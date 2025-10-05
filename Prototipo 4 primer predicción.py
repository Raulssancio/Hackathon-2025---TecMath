import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import os

# ---------------------- Configuración ----------------------
COLOR_FONDO = "#f0f4f8"
COLOR_BOTON = "#4CAF50"
COLOR_BOTON_SEC = "#2196F3"
COLOR_TEXTO = "#333333"
FUENTE_TITULO = ("Segoe UI", 16, "bold")
FUENTE_TEXTO = ("Segoe UI", 12)

# Lista de ciudades
ciudades = ["AIQ - Aeropuerto de Querétaro"]

# Carpeta donde está el script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# GIFs
GIFS = {
    "Frío": os.path.join(SCRIPT_DIR, "frio.gif"),
    "Templado": os.path.join(SCRIPT_DIR, "templado.gif"),
    "Caluroso": os.path.join(SCRIPT_DIR, "caluroso.gif"),
    "Muy caluroso": os.path.join(SCRIPT_DIR, "muy_caluroso.gif")
}

# ---------------------- Función para generar CSV diario ----------------------
def generar_csv_diario(temp_csv):
    temp_df = pd.read_csv(temp_csv)
    temp_df.rename(columns={'FechaHora': 'FechaHora', 'Temp': 'Temp'}, inplace=True)

    # Convertir a datetime
    temp_df['FechaHora'] = pd.to_datetime(temp_df['FechaHora'])

    # Ajuste de 12 horas (si tu CSV requiere)
    temp_df['FechaHora'] = temp_df['FechaHora'] + pd.Timedelta(hours=12)

    # Conversión Kelvin -> Celsius
    temp_df['Temp'] = temp_df['Temp'] - 273.15

    # Agregar columnas auxiliares
    temp_df['Year'] = temp_df['FechaHora'].dt.year
    temp_df['Month'] = temp_df['FechaHora'].dt.month
    temp_df['Day'] = temp_df['FechaHora'].dt.day
    temp_df['Hour'] = temp_df['FechaHora'].dt.hour

    return temp_df

# ---------------------- Cargar CSV ----------------------
df_temp = generar_csv_diario("Temperatura del aire cerca del suelo.csv")

# ---------------------- Función para etiquetas y gifs ----------------------
def etiqueta_gif(temp):
    if temp < 10:
        return "Frío", GIFS["Frío"]
    elif temp < 20:
        return "Templado", GIFS["Templado"]
    elif temp < 30:
        return "Caluroso", GIFS["Caluroso"]
    else:
        return "Muy caluroso", GIFS["Muy caluroso"]

# ---------------------- Función regresión lineal ----------------------
def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    m = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x**2) - (np.sum(x))**2)
    b = (np.sum(y) - m*np.sum(x)) / n
    return m, b

# ---------------------- Funciones Tkinter ----------------------
def abrir_temperaturas_dia():
    temp_win = tk.Toplevel(root)
    temp_win.title("Temperaturas del Día")
    temp_win.geometry("600x500")
    temp_win.configure(bg=COLOR_FONDO)

    tk.Label(temp_win, text="Temperaturas del día", font=FUENTE_TITULO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=15)

    tk.Label(temp_win, text="Ciudad: AIQ - Aeropuerto de Querétaro", font=FUENTE_TEXTO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=5)

    tk.Label(temp_win, text="Selecciona la fecha:", font=FUENTE_TEXTO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=10)

    fecha_entry = DateEntry(temp_win, width=12)
    fecha_entry.pack(pady=5)

    resultado = tk.Label(temp_win, text="", font=FUENTE_TEXTO, bg=COLOR_FONDO, justify="left")
    resultado.pack(pady=10)

    gif_label = tk.Label(temp_win, bg=COLOR_FONDO)
    gif_label.pack(pady=10)

    def mostrar_temperaturas():
        fecha_sel = fecha_entry.get_date()
        df_dia = df_temp[(df_temp['Month'] == fecha_sel.month) &
                         (df_temp['Day'] == fecha_sel.day)]

        if df_dia.empty:
            resultado.config(text="No hay datos para esa fecha.")
            gif_label.config(image="")
            return

        # Corregir desfase de horas por cambio horario
        df_dia['Hour'] = ((df_dia['Hour'] // 3) * 3)  # 0,3,6,...
        df_dia_resample = df_dia.groupby('Hour')['Temp'].mean().reset_index()

        texto = f"Temperaturas para el {fecha_sel}:\n\n"
        for idx, row in df_dia_resample.iterrows():
            hora = int(row['Hour'])
            temp = row['Temp']
            texto += f"{hora:02d}:00 - {temp:.1f} °C\n"

        texto += f"\nMáxima: {df_dia['Temp'].max():.1f} °C"
        texto += f"\nMínima: {df_dia['Temp'].min():.1f} °C"

        resultado.config(text=texto)

        # Mostrar gif de la última temperatura promedio
        etiqueta, gif_path = etiqueta_gif(df_dia['Temp'].mean())
        try:
            gif_img = Image.open(gif_path)
            gif_img = gif_img.resize((150,150))
            gif_img_tk = ImageTk.PhotoImage(gif_img)
            gif_label.image = gif_img_tk
            gif_label.config(image=gif_img_tk)
        except FileNotFoundError:
            gif_label.config(image="")

    tk.Button(temp_win, text="Mostrar temperaturas", command=mostrar_temperaturas,
              font=FUENTE_TEXTO, bg=COLOR_BOTON, fg="white", relief="flat").pack(pady=15)

def predecir_temperaturas():
    pred_win = tk.Toplevel(root)
    pred_win.title("Predicción de Temperaturas")
    pred_win.geometry("600x500")
    pred_win.configure(bg=COLOR_FONDO)

    tk.Label(pred_win, text="Predicción de Temperaturas", font=FUENTE_TITULO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=15)

    tk.Label(pred_win, text="Selecciona la fecha:", font=FUENTE_TEXTO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=10)

    fecha_entry = DateEntry(pred_win, width=12)
    fecha_entry.pack(pady=5)

    resultado = tk.Label(pred_win, text="", font=FUENTE_TEXTO, bg=COLOR_FONDO, justify="left")
    resultado.pack(pady=10)

    gif_label = tk.Label(pred_win, bg=COLOR_FONDO)
    gif_label.pack(pady=10)

    def predecir():
        fecha_sel = fecha_entry.get_date()
        year_sel = fecha_sel.year

        # Filtrar datos históricos del mismo día y mes
        df_filtrado = df_temp[(df_temp['Month'] == fecha_sel.month) &
                              (df_temp['Day'] == fecha_sel.day) &
                              (df_temp['Year'] < year_sel)].copy()

        if df_filtrado.empty:
            resultado.config(text="No hay datos históricos para esa fecha.")
            gif_label.config(image="")
            return

        # Ajustar horas a múltiplos de 3
        df_filtrado['Hour'] = ((df_filtrado['Hour'] // 3) * 3)

        df_resample = df_filtrado.groupby(['Hour', 'Year'])['Temp'].mean().reset_index()

        # Predecir temperatura usando regresión lineal para cada hora
        texto = f"Predicción para {fecha_sel}:\n\n"
        for h in sorted(df_resample['Hour'].unique()):
            y = df_resample[df_resample['Hour'] == h]['Temp'].values
            x = np.arange(len(y))
            m, b = linear_regression(x, y)
            anio_index = year_sel - 2015
            temp_pred = m * anio_index + b
            texto += f"{h:02d}:00 - {temp_pred:.1f} °C\n"

        # Mostrar predicción máxima y mínima
        temps_pred = [float(t.split("-")[1].replace("°C","").strip()) for t in texto.split("\n")[2:-1]]
        texto += f"\nMáxima predicha: {max(temps_pred):.1f} °C"
        texto += f"\nMínima predicha: {min(temps_pred):.1f} °C"

        resultado.config(text=texto)

        # Mostrar gif de la temperatura promedio
        avg_temp_pred = np.mean(temps_pred)
        etiqueta, gif_path = etiqueta_gif(avg_temp_pred)
        try:
            gif_img = Image.open(gif_path)
            gif_img = gif_img.resize((150,150))
            gif_img_tk = ImageTk.PhotoImage(gif_img)
            gif_label.image = gif_img_tk
            gif_label.config(image=gif_img_tk)
        except FileNotFoundError:
            gif_label.config(image="")

    tk.Button(pred_win, text="Predecir temperaturas", command=predecir,
              font=FUENTE_TEXTO, bg=COLOR_BOTON_SEC, fg="white", relief="flat").pack(pady=15)

# ---------------------- Ventana Principal ----------------------
root = tk.Tk()
root.title("App del Clima - AIQ")
root.geometry("420x300")
root.configure(bg=COLOR_FONDO)

titulo = tk.Label(root, text="☁️ App del Clima AIQ ☁️", font=FUENTE_TITULO,
                  bg=COLOR_FONDO, fg=COLOR_TEXTO)
titulo.pack(pady=20)

btn_temp_dia = tk.Button(root, text="Ver temperaturas del día", command=abrir_temperaturas_dia,
                         font=FUENTE_TEXTO, bg=COLOR_BOTON, fg="white",
                         relief="flat", padx=10, pady=5, width=25)
btn_temp_dia.pack(pady=10)

btn_prediccion = tk.Button(root, text="Predecir temperaturas", command=predecir_temperaturas,
                           font=FUENTE_TEXTO, bg=COLOR_BOTON_SEC, fg="white",
                           relief="flat", padx=10, pady=5, width=25)
btn_prediccion.pack(pady=10)

root.mainloop()




