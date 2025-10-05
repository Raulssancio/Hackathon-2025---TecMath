# cowdy_weather.py
# Requisitos: pandas, numpy, pillow, scikit-learn, tkcalendar
# Coloca este .py en la misma carpeta que tus CSVs y gifs (frio.gif, templado.gif, caluroso.gif, muy_caluroso.gif)

import os
import tkinter as tk
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from itertools import count
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ---------------------- Config ----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILES = {
    "air_temp": ("Temperatura del aire cerca del suelo.csv", "FechaHora", "Temp"),
    "soil_t_0_10": ("Temperatura del suelo (0 - 10 cm bajo tierra).csv", "FechaHora", "Temp_Suelo_0_10"),
    "soil_t_10_40": ("Temperatura del suelo (10 - 40 cm bajo tierra).csv", "FechaHora", "Temp_Suelo_10_40"),
    "soil_h_0_10": ("Humedad del suelo (0 - 10 cm bajo tierra).csv", "FechaHora", "Humedad_Suelo_0_10"),
    "evap": ("Evaporación directa del suelo.csv", "FechaHora", "Evaporacion"),
    "pres": ("Presión del aire al nivel del suelo.csv", "FechaHora", "Presion"),
    "wind": ("Velocidad del viento en la superficie, promedio en el tiempo.csv", "FechaHora", "Viento"),
    "prec": ("Tasa de precipitacion de lluvia.csv", "FechaHora", "Precipitacion"),
    "heat_flux": ("Flujo de calor sensible.csv", "FechaHora", "Flujo_Calor"),
}

# Flip de CSVs por bandera
FLIP_HOURS = {
    "air_temp": True,   # invertir si las temperaturas salen volteadas
    "soil_t_0_10": False,
    "soil_t_10_40": False,
    "soil_h_0_10": False,
    "evap": False,
    "pres": False,
    "wind": False,
    "prec": False,
    "heat_flux": False
}

GIFS = {
    "frío": os.path.join(SCRIPT_DIR, "frio.gif"),
    "templado": os.path.join(SCRIPT_DIR, "templado.gif"),
    "caluroso": os.path.join(SCRIPT_DIR, "caluroso.gif"),
    "muy caluroso": os.path.join(SCRIPT_DIR, "muy_caluroso.gif")
}

# UI colors / fonts
COLOR_FONDO = "#f0f4f8"
COLOR_BOTON = "#4CAF50"
COLOR_BOTON_SEC = "#2196F3"
COLOR_TEXTO = "#333333"
FUENTE_TITULO = ("Segoe UI", 16, "bold")
FUENTE_TEXTO = ("Segoe UI", 12)

# ---------------------- Gif animado ----------------------
def mostrar_gif_animado(label, gif_path, size=(150,150), delay_ms=25):
    try:
        im = Image.open(gif_path)
    except Exception:
        label.config(image="")
        return
    frames = []
    try:
        for i in count(0):
            im.seek(i)
            frame = im.copy().resize(size, Image.Resampling.LANCZOS)
            frames.append(ImageTk.PhotoImage(frame))
    except EOFError:
        pass
    if not frames:
        label.config(image="")
        return
    def update(ind=0):
        frame = frames[ind]
        label.configure(image=frame)
        label.image = frame
        label.after(delay_ms, update, (ind+1) % len(frames))
    update()

# ---------------------- Load single CSV ----------------------
def load_csv(filepath, time_col, value_col, flip=False):
    if not os.path.exists(filepath):
        print(f"[WARN] No existe: {filepath}")
        return None
    df = pd.read_csv(filepath)
    df = df.rename(columns={time_col: "FechaHora", value_col: value_col})
    df['FechaHora'] = pd.to_datetime(df['FechaHora'], errors='coerce')
    df = df.dropna(subset=['FechaHora'])
    
    # Aplicar flip si se requiere
    if flip:
        df = df.iloc[::-1].reset_index(drop=True)
    
    # ----- CORRECCIÓN DE DESFASE HORARIO SOLO PARA TEMP AIRE -----
    if value_col == "Temp":
        df['FechaHora'] = df['FechaHora'] + pd.Timedelta(hours=12)  # corrige desfase de 12h
    
    df['FechaHora_3h'] = df['FechaHora'].dt.floor('3h')
    df = df.groupby('FechaHora_3h', as_index=False)[value_col].mean()
    return df

# ---------------------- Build master dataframe ----------------------
def build_master_df():
    dfs = {}
    for key, (fname, time_col, val_col) in CSV_FILES.items():
        path = os.path.join(SCRIPT_DIR, fname)
        df = load_csv(path, time_col, val_col, flip=FLIP_HOURS.get(key, False))
        if df is not None:
            dfs[key] = df
    # Merge
    if 'air_temp' in dfs:
        master = dfs['air_temp'].rename(columns={'Temp': 'Temp_air'}).copy()
    else:
        all_times = pd.concat([d[['FechaHora_3h']] for d in dfs.values()]).drop_duplicates().sort_values('FechaHora_3h')
        master = all_times.reset_index(drop=True)
    for key, df in dfs.items():
        if key == 'air_temp':
            continue
        val_col = df.columns[1]
        master = master.merge(df, on='FechaHora_3h', how='left')
    if 'Temp_air' in master.columns:
        master = master.rename(columns={'Temp_air': 'Temp'})
    master = master.rename(columns={'FechaHora_3h': 'FechaHora'})
    master['Year'] = master['FechaHora'].dt.year
    master['Month'] = master['FechaHora'].dt.month
    master['Day'] = master['FechaHora'].dt.day
    master['Hour'] = master['FechaHora'].dt.hour
    return master

# ---------------------- Prepare & train models ----------------------
def prepare_and_train(master_df):
    feature_cols = [
        'Temp_Suelo_0_10', 'Temp_Suelo_10_40', 'Humedad_Suelo_0_10',
        'Evaporacion', 'Presion', 'Viento', 'Precipitacion', 'Flujo_Calor'
    ]
    for c in feature_cols:
        if c not in master_df.columns:
            master_df[c] = np.nan
    # Temperatura en Celsius
    if 'Temp' in master_df.columns and master_df['Temp'].dropna().mean() > 200:
        master_df['Temp'] = master_df['Temp'] - 273.15
    # Impute missing
    impute_vals = {}
    for c in feature_cols:
        impute_vals[c] = master_df[c].mean(skipna=True)
        master_df[c] = master_df[c].fillna(impute_vals[c])
    # Modelo Temp
    train_temp = master_df.dropna(subset=['Temp']).copy()
    X_temp = train_temp[feature_cols].values
    y_temp = train_temp['Temp'].values
    model_temp = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
    model_temp.fit(X_temp, y_temp)
    print("[INFO] Modelo de temperatura entrenado con", len(train_temp), "filas.")
    # Modelo Precipitación
    train_prec = master_df.dropna(subset=['Precipitacion']).copy()
    X_prec = train_prec[feature_cols].values
    y_prec = train_prec['Precipitacion'].values
    model_prec = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
    model_prec.fit(X_prec, y_prec)
    print("[INFO] Modelo de lluvia entrenado con", len(train_prec), "filas.")
    return model_temp, model_prec, feature_cols, impute_vals

# ---------------------- Predict for date ----------------------
def predict_for_date(model, master_df, feature_cols, fecha_sel, target='Temp'):
    preds = []
    for hour in range(0,24,3):
        hist = master_df[
            (master_df['Month']==fecha_sel.month) &
            (master_df['Day']==fecha_sel.day) &
            (master_df['Hour']==hour) &
            (master_df['Year']<fecha_sel.year)
        ]
        if hist.empty:
            hist = master_df[(master_df['Hour']==hour)]
        if hist.empty:
            feat_vec = np.array([master_df[c].mean() for c in feature_cols]).reshape(1,-1)
        else:
            feat_vec = np.array([hist[c].mean() for c in feature_cols]).reshape(1,-1)
        pred = float(model.predict(feat_vec)[0])
        preds.append((hour, pred))
    return preds

# ---------------------- Utilities ----------------------
def etiqueta_por_temp(t):
    if t < 10: return "frío"
    elif t < 20: return "templado"
    elif t < 30: return "caluroso"
    else: return "muy caluroso"

# ---------------------- Main ----------------------
print("[INFO] Construyendo dataset maestro y entrenando modelos...")
master_df = build_master_df()
model_temp, model_prec, feature_cols, impute_vals = prepare_and_train(master_df)

# ---------------------- Tkinter UI ----------------------
root = tk.Tk()
root.title("Cowdy Weather 🌤️🐮")
root.geometry("500x450")
root.configure(bg=COLOR_FONDO)

tk.Label(root, text="Cowdy Weather 🌤️🐮", font=FUENTE_TITULO, bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=14)

frame = tk.Frame(root, bg=COLOR_FONDO)
frame.pack(pady=10)
tk.Label(frame, text="Selecciona fecha:", font=FUENTE_TEXTO, bg=COLOR_FONDO).grid(row=0,column=0,padx=6)
fecha_entry = DateEntry(frame, width=12)
fecha_entry.grid(row=0,column=1,padx=6)
btn_pred = tk.Button(frame, text="Predecir", bg=COLOR_BOTON_SEC, fg="white", font=FUENTE_TEXTO)
btn_pred.grid(row=0,column=2,padx=8)

resultado = tk.Label(root, text="", font=FUENTE_TEXTO, bg=COLOR_FONDO, justify="left")
resultado.pack(pady=8)
gif_label = tk.Label(root, bg=COLOR_FONDO)
gif_label.pack(pady=6)

def on_predict_click():
    fecha_sel = fecha_entry.get_date()
    # Temp
    preds_temp = predict_for_date(model_temp, master_df, feature_cols, fecha_sel, target='Temp')
    texto = f"Predicción de temperatura para {fecha_sel} (cada 3h):\n\n"
    temps = []
    for h,t in preds_temp:
        texto += f"{h:02d}:00 — {t:.1f} °C\n"
        temps.append(t)
    texto += f"\nMáx: {max(temps):.1f} °C  |  Mín: {min(temps):.1f} °C  |  Prom: {np.mean(temps):.1f} °C\n"
    etiqueta = etiqueta_por_temp(np.mean(temps))
    mostrar_gif_animado(gif_label, GIFS.get(etiqueta), size=(160,160), delay_ms=25)
    # Precipitación %
    preds_prec = predict_for_date(model_prec, master_df, feature_cols, fecha_sel, target='Precipitacion')
    texto += "\nProbabilidad de lluvia estimada:\n"
    probs = []
    umbral_lluvia = 0.0003
    for h,p in preds_prec:
        prob = min(100,(p/umbral_lluvia)*100)
        texto += f"{h:02d}:00 — {prob:.0f} %\n"
        probs.append(prob)
    texto += f"\nPromedio del día: {np.mean(probs):.0f} %"
    resultado.config(text=texto)

btn_pred.config(command=on_predict_click)

info = tk.Label(root, text="Modelos entrenados automáticamente (Ridge). Usa los CSV en la misma carpeta.", font=("Segoe UI",10), bg=COLOR_FONDO)
info.pack(pady=6)

root.mainloop()







