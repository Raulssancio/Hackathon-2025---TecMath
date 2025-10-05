# cowdy_weather_flet_bandas_fix.py — Flet 0.22+ SIN scikit-learn
# Requiere: pandas, numpy, pillow, flet
# CSVs: ver CSV_FILES. GIFs: frio.gif, templado.gif, caluroso.gif, muy_caluroso.gif, lluvia.gif

import os
import base64
import flet as ft
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

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

# Si un CSV viene invertido en el tiempo, pon True
FLIP_HOURS = {
    "air_temp": True,
    "soil_t_0_10": False,
    "soil_t_10_40": False,
    "soil_h_0_10": False,
    "evap": False,
    "pres": False,
    "wind": False,
    "prec": False,
    "heat_flux": False,
}

GIF_PATHS = {
    "frío": os.path.join(SCRIPT_DIR, "frio.gif"),
    "templado": os.path.join(SCRIPT_DIR, "templado.gif"),
    "caluroso": os.path.join(SCRIPT_DIR, "caluroso.gif"),
    "muy caluroso": os.path.join(SCRIPT_DIR, "muy_caluroso.gif"),
    "lluvia": os.path.join(SCRIPT_DIR, "lluvia.gif"),
}

# UI
COLOR_FONDO = "#f0f4f8"
COLOR_TEXTO = "#333333"
COLOR_BTN = "#4CAF50"
COLOR_BTN_SEC = "#2196F3"
CIUDAD_FIJA = "AIQ - Aeropuerto de Querétaro"

# Lluvia
UMBRAL_LLUVIA = 0.0003     # tasa de precipitación (referencia para escalar a prob.)
UMBRAL_PROB_DEF = 0.50     # umbral prob si no se puede calibrar
MEAN_PROB_THRESHOLD = 0.20 # media diaria mínima (20%) para considerar "llueve"

# ---------------------- Utilidades ----------------------
def to_date_any(v):
    """Convierte (date | datetime | str) → datetime.date o None."""
    try:
        if isinstance(v, date) and not isinstance(v, datetime):
            return v
        return pd.to_datetime(str(v)).date()
    except Exception:
        return None

def gif_to_b64(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def etiqueta_por_temp(t):
    if t <= 16: return "frío"
    elif t <= 25: return "templado"
    elif t <= 35: return "caluroso"
    else: return "muy caluroso"

# ---------------------- Mini Ridge (sin sklearn) ----------------------
class RidgeReg:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.mean_ = None
        self.std_ = None
        self.coef_ = None
        self.intercept_ = None
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Z = (X - self.mean_) / self.std_
        n, p = Z.shape
        Z1 = np.hstack([Z, np.ones((n, 1))])
        A = Z1.T @ Z1
        A[:p, :p] += self.alpha * np.eye(p)
        b = Z1.T @ y
        beta = np.linalg.solve(A, b)
        self.coef_ = beta[:p]
        self.intercept_ = beta[p]
        return self
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Z = (X - self.mean_) / self.std_
        return Z @ self.coef_ + self.intercept_

# ---------------------- Carga CSVs (a 3 horas) ----------------------
def load_csv(filepath, time_col, value_col, flip=False):
    if not os.path.exists(filepath):
        print(f"[WARN] No existe: {filepath}")
        return None
    df = pd.read_csv(filepath)
    df = df.rename(columns={time_col: "FechaHora", value_col: value_col})
    df["FechaHora"] = pd.to_datetime(df["FechaHora"], errors="coerce")
    df = df.dropna(subset=["FechaHora"])
    if flip:
        df = df.iloc[::-1].reset_index(drop=True)
    if value_col == "Temp":
        df["FechaHora"] = df["FechaHora"] + pd.Timedelta(hours=12)
    df["FechaHora_3h"] = df["FechaHora"].dt.floor("3h")
    df = df.groupby("FechaHora_3h", as_index=False)[value_col].mean()
    return df

def build_master_df():
    dfs = {}
    for key, (fname, time_col, val_col) in CSV_FILES.items():
        path = os.path.join(SCRIPT_DIR, fname)
        df = load_csv(path, time_col, val_col, flip=FLIP_HOURS.get(key, False))
        if df is not None:
            dfs[key] = df
    if not dfs:
        raise RuntimeError("No se pudo cargar ningún CSV.")

    if "air_temp" in dfs:
        master = dfs["air_temp"].rename(columns={"Temp": "Temp_air"}).copy()
    else:
        all_times = pd.concat([d[["FechaHora_3h"]] for d in dfs.values()]).drop_duplicates().sort_values("FechaHora_3h")
        master = all_times.reset_index(drop=True)

    for key, dfk in dfs.items():
        if key == "air_temp":
            continue
        val_col = dfk.columns[1]
        master = master.merge(dfk, on="FechaHora_3h", how="left")

    if "Temp_air" in master.columns:
        master = master.rename(columns={"Temp_air": "Temp"})
    master = master.rename(columns={"FechaHora_3h": "FechaHora"})

    master["Year"]  = master["FechaHora"].dt.year
    master["Month"] = master["FechaHora"].dt.month
    master["Day"]   = master["FechaHora"].dt.day
    master["Hour"]  = master["FechaHora"].dt.hour
    return master

# ---------------------- Entrenamiento & calibración ----------------------
def prepare_and_train(master_df):
    feature_cols = [
        "Temp_Suelo_0_10", "Temp_Suelo_10_40", "Humedad_Suelo_0_10",
        "Evaporacion", "Presion", "Viento", "Precipitacion", "Flujo_Calor",
    ]
    for c in feature_cols:
        if c not in master_df.columns:
            master_df[c] = np.nan
    if "Temp" in master_df.columns and master_df["Temp"].dropna().mean() > 200:
        master_df["Temp"] = master_df["Temp"] - 273.15
    impute_vals = {}
    for c in feature_cols:
        impute_vals[c] = float(master_df[c].mean(skipna=True))
        master_df[c] = master_df[c].fillna(impute_vals[c])

    train_temp = master_df.dropna(subset=["Temp"]).copy()
    X_temp = train_temp[feature_cols].values
    y_temp = train_temp["Temp"].values
    model_temp = RidgeReg(alpha=1.0).fit(X_temp, y_temp)
    print("[INFO] Modelo Temp (sin sklearn) entrenado con", len(train_temp), "filas.")

    if "Precipitacion" in master_df.columns:
        train_prec = master_df.dropna(subset=["Precipitacion"]).copy()
        X_prec = train_prec[feature_cols].values
        y_prec = train_prec["Precipitacion"].values
        model_prec = RidgeReg(alpha=1.0).fit(X_prec, y_prec)
        print("[INFO] Modelo Lluvia (sin sklearn) entrenado con", len(train_prec), "filas.")
    else:
        model_prec = None
        print("[WARN] No hay columna 'Precipitacion' para entrenar.")

    return model_temp, model_prec, feature_cols, impute_vals

def calibrar_umbral_prob(master_df, model_prec, feature_cols, umbral_lluvia=UMBRAL_LLUVIA):
    if model_prec is None or "Precipitacion" not in master_df.columns:
        return UMBRAL_PROB_DEF
    dfc = master_df.dropna(subset=["Precipitacion"]).copy()
    if dfc.empty:
        return UMBRAL_PROB_DEF

    X = dfc[feature_cols].values
    y_true = (dfc["Precipitacion"] >= umbral_lluvia).astype(int).values
    y_rate = model_prec.predict(X)
    y_prob = np.clip(y_rate / umbral_lluvia, 0.0, 1.0)

    thresholds = np.linspace(0.1, 0.9, 17)
    best_f1, best_t = -1.0, UMBRAL_PROB_DEF
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    print(f"[INFO] Umbral por hora calibrado = {best_t:.2f} (F1={best_f1:.2f})")
    return best_t

# ---------------------- Predicciones auxiliares ----------------------
def predict_for_date(model, master_df, feature_cols, fecha_sel):
    preds = []
    for hour in range(0, 24, 3):
        hist = master_df[
            (master_df["Month"] == fecha_sel.month)
            & (master_df["Day"] == fecha_sel.day)
            & (master_df["Hour"] == hour)
            & (master_df["Year"] < fecha_sel.year)
        ]
        if hist.empty:
            hist = master_df[(master_df["Hour"] == hour)]
        if hist.empty:
            feat_vec = np.array([master_df[c].mean() for c in feature_cols]).reshape(1, -1)
        else:
            feat_vec = np.array([hist[c].mean() for c in feature_cols]).reshape(1, -1)
        pred = float(model.predict(feat_vec)[0])
        preds.append((hour, pred))
    return preds

def mean_prob_for_date(model_prec, master_df, feature_cols, fecha_sel):
    if model_prec is None:
        return [], None
    preds_prec = predict_for_date(model_prec, master_df, feature_cols, fecha_sel)
    probs01 = [float(np.clip(p_rate / UMBRAL_LLUVIA, 0.0, 1.0)) for _, p_rate in preds_prec]
    return probs01, (np.mean(probs01) if probs01 else None)

def mean_temp_for_date(model_temp, master_df, feature_cols, fecha_sel):
    preds_temp = predict_for_date(model_temp, master_df, feature_cols, fecha_sel)
    temps = [t for _, t in preds_temp]
    return temps, (np.mean(temps) if temps else None)

def find_nearest_rain_day(model_prec, master_df, feature_cols, base_date, mean_threshold=MEAN_PROB_THRESHOLD, max_days=365):
    if model_prec is None:
        return None, None
    base_date = to_date_any(base_date)
    if not base_date:
        return None, None
    for d in range(0, max_days + 1):
        for sign in ([0] if d == 0 else [-1, 1]):
            target = base_date + sign * timedelta(days=d)
            _, meanp = mean_prob_for_date(model_prec, master_df, feature_cols, target)
            if meanp is not None and meanp >= mean_threshold:
                return target, meanp
    return None, None

# ---------- Bandas exclusivas de temperatura media ----------
def find_nearest_temp_day_in_band(
    model_temp, master_df, feature_cols, base_date,
    min_c=None, max_c=None, inclusive_max=False, max_days=365
):
    """
    Busca el día más cercano donde la TEMPERATURA MEDIA esté en la banda:
      [min_c, max_c) por defecto.
      Si inclusive_max=True, entonces [min_c, max_c] (incluye el límite superior).
      Si min_c es None → (-inf, max_c)
      Si max_c es None → [min_c, +inf)
    """
    base_date = to_date_any(base_date)
    if not base_date:
        return None, None

    for d in range(0, max_days + 1):
        for sign in ([0] if d == 0 else [-1, 1]):
            target = base_date + sign * timedelta(days=d)
            _, mean_t = mean_temp_for_date(model_temp, master_df, feature_cols, target)
            if mean_t is None:
                continue

            lo_ok = True if min_c is None else (mean_t >= min_c)
            if max_c is None:
                hi_ok = True
            else:
                hi_ok = (mean_t <= max_c + 1e-6) if inclusive_max else (mean_t < max_c)

            if lo_ok and hi_ok:
                return target, mean_t
    return None, None

# ---------------------- App Flet ----------------------
def main(page: ft.Page):
    page.title = "Cowdy Weather 🌤️🐮 "
    page.bgcolor = COLOR_FONDO
    page.padding = 16
    page.scroll = True

    titulo = ft.Text("Cowdy Weather 🌤️🐮", size=22, weight=ft.FontWeight.BOLD, color=COLOR_TEXTO)
    subtitulo = ft.Text(f"Ciudad: {CIUDAD_FIJA}", size=14, color=COLOR_TEXTO)
    info_modelo = ft.Text("", size=12, color=COLOR_TEXTO)
    info_umbral = ft.Text("", size=12, color=COLOR_TEXTO)
    resultado = ft.Text("", size=16, color=COLOR_TEXTO)
    imagen = ft.Image(src="", src_base64=None, width=220, height=220, fit=ft.ImageFit.CONTAIN)

    # Cargar y entrenar
    try:
        master_df = build_master_df()
        model_temp, model_prec, feature_cols, impute_vals = prepare_and_train(master_df)
        n_filas = len(master_df)
        rango = (master_df["FechaHora"].min(), master_df["FechaHora"].max())
        info_modelo.value = f"Dataset maestro: {n_filas} filas • Rango: {rango[0]} → {rango[1]}"
        umbral_prob_opt = calibrar_umbral_prob(master_df, model_prec, feature_cols, UMBRAL_LLUVIA)
        info_umbral.value = f"Umbral por hora calibrado: {umbral_prob_opt:.2f} • Umbral media lluvia: {MEAN_PROB_THRESHOLD:.2f} • (tasa ref={UMBRAL_LLUVIA})"
    except Exception as e:
        page.add(titulo, ft.Divider(), ft.Text(f"⚠️ Error cargando/entrenando: {e}", color="red"))
        return

    # DatePicker
    dp = ft.DatePicker()
    page.overlay.append(dp)
    txt_fecha = ft.TextField(label="Fecha seleccionada", read_only=True, width=200)

    def set_gif(avg_temp: float, probs01):
        # Lluvia si: (alguna hora ≥ umbral calibrado) o (media ≥ 0.20)
        lluvia = False
        if probs01:
            if (any(p >= umbral_prob_opt for p in probs01)) or (np.mean(probs01) >= MEAN_PROB_THRESHOLD):
                lluvia = True

        imagen.src = ""
        imagen.src_base64 = None
        page.update()

        if lluvia:
            b64 = gif_to_b64(GIF_PATHS.get("lluvia"))
        else:
            etiqueta = etiqueta_por_temp(avg_temp)
            b64 = gif_to_b64(GIF_PATHS.get(etiqueta))

        if b64:
            imagen.src_base64 = b64
        page.update()

    def predecir_para_fecha(fecha_sel_date: date):
        # Temperatura cada 3h
        preds_temp = predict_for_date(model_temp, master_df, feature_cols, fecha_sel_date)
        temps = [t for _, t in preds_temp]

        texto = f"Predicción de temperatura para {fecha_sel_date} (cada 3h):\n\n"
        for h, t in preds_temp:
            texto += f"{h:02d}:00 — {t:.1f} °C\n"
        texto += f"\nMáx: {max(temps):.1f} °C  |  Mín: {min(temps):.1f} °C  |  Prom: {np.mean(temps):.1f} °C\n"

        probs01 = []
        if model_prec is not None:
            preds_prec = predict_for_date(model_prec, master_df, feature_cols, fecha_sel_date)
            texto += "\nProbabilidad de lluvia estimada:\n"
            for h, p_rate in preds_prec:
                prob = float(np.clip(p_rate / UMBRAL_LLUVIA, 0.0, 1.0))
                probs01.append(prob)
                texto += f"{h:02d}:00 — {prob*100:.0f} %\n"
            if probs01:
                meanp = np.mean(probs01)
                texto += f"\nPromedio del día: {meanp*100:.0f} %"
                lluvia_flag = (any(p >= umbral_prob_opt for p in probs01)) or (meanp >= MEAN_PROB_THRESHOLD)
                texto += f"\nDecisión (hora≥{umbral_prob_opt:.2f}  o  media≥{MEAN_PROB_THRESHOLD:.2f}): "
                texto += "🌧️ Probabilidad de lluvia" if lluvia_flag else "🌤️ NO llueve"

        set_gif(float(np.mean(temps)), probs01)
        resultado.value = texto
        page.update()

    # --- Buscar día cercano con lluvia (media ≥ 20%) ---
    def on_buscar_cercano_lluvia(_):
        base = to_date_any(dp.value) or to_date_any(txt_fecha.value) or datetime.today().date()
        dia, meanp = find_nearest_rain_day(model_prec, master_df, feature_cols, base, MEAN_PROB_THRESHOLD, max_days=365)
        if not dia:
            page.snack_bar = ft.SnackBar(ft.Text("No se encontró un día cercano con media de lluvia ≥ 20% en ±365 días."))
            page.snack_bar.open = True
            page.update()
            return
        dp.value = dia
        txt_fecha.value = str(dia)
        page.update()
        predecir_para_fecha(dia)

    # --- Handlers: bandas de temperatura (incluyendo 25–≤35) ---
    def temp_band_handler(min_c, max_c, etiqueta_btn, inclusive_max=False):
        def _handler(_):
            base = to_date_any(dp.value) or to_date_any(txt_fecha.value) or datetime.today().date()
            dia, mean_t = find_nearest_temp_day_in_band(
                model_temp, master_df, feature_cols, base,
                min_c=min_c, max_c=max_c, inclusive_max=inclusive_max, max_days=365
            )
            if not dia:
                if min_c is None:
                    rango = f"T < {max_c:.0f}°C"
                elif max_c is None:
                    rango = f"T ≥ {min_c:.0f}°C"
                else:
                    rango = f"{min_c:.0f}°C ≤ T ≤ {max_c:.0f}°C" if inclusive_max else f"{min_c:.0f}°C ≤ T < {max_c:.0f}°C"
                page.snack_bar = ft.SnackBar(ft.Text(f"No se encontró un día cercano con {rango} en ±365 días."))
                page.snack_bar.open = True
                page.update()
                return
            dp.value = dia
            txt_fecha.value = str(dia)
            page.update()
            predecir_para_fecha(dia)
        return _handler

    # Handlers de fecha
    def abrir_dp(_):
        dp.open = True
        page.update()

    def on_dp_change(e: ft.ControlEvent):
        fsel = to_date_any(dp.value)
        if not fsel:
            return
        txt_fecha.value = str(fsel)
        page.update()
        predecir_para_fecha(fsel)

    dp.on_change = on_dp_change

    def on_predecir(_):
        fsel = to_date_any(dp.value) or to_date_any(datetime.today().date())
        txt_fecha.value = str(fsel)
        page.update()
        predecir_para_fecha(fsel)

    # Botones
    btn_elegir = ft.FilledButton(
        "Elegir fecha",
        icon=ft.Icons.EVENT,
        style=ft.ButtonStyle(bgcolor=COLOR_BTN_SEC, color=ft.Colors.WHITE),
        on_click=abrir_dp,
    )
    btn_predecir = ft.FilledButton(
        "Predecir",
        icon=ft.Icons.SHOW_CHART,
        style=ft.ButtonStyle(bgcolor=COLOR_BTN, color=ft.Colors.WHITE),
        on_click=on_predecir,
    )
    btn_buscar_lluvia = ft.FilledButton(
        "Día cercano con lluvia (≥ media 20%)",
        icon=ft.Icons.NEAR_ME,
        on_click=on_buscar_cercano_lluvia,
    )
    # Bandas exclusivas:
    btn_t_lt16  = ft.FilledButton("Día cercano T<16°C",
                                  icon=ft.Icons.DEVICE_THERMOSTAT,
                                  on_click=temp_band_handler(None, 16.0, "T<16"))
    btn_t_16_25 = ft.FilledButton("Día cercano 16–<25°C",
                                  icon=ft.Icons.DEVICE_THERMOSTAT,
                                  on_click=temp_band_handler(16.0, 25.0, "16-<25"))
    btn_t_25_35 = ft.FilledButton("Día cercano 25–≤35°C",
                                  icon=ft.Icons.DEVICE_THERMOSTAT,
                                  on_click=temp_band_handler(25.0, 35.0, "25-≤35", inclusive_max=True))

    # Layout
    page.add(
        ft.Container(
            content=ft.Column(
                [
                    titulo,
                    subtitulo,
                    info_modelo,
                    info_umbral,
                    ft.Divider(),
                    ft.Row(
                        [btn_elegir, txt_fecha, btn_predecir, btn_buscar_lluvia],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        wrap=True,
                    ),
                    ft.Row(
                        [btn_t_lt16, btn_t_16_25, btn_t_25_35],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        wrap=True,
                    ),
                    ft.Divider(),
                    resultado,
                    imagen,
                ],
                spacing=12,
            ),
            padding=16,
            bgcolor=ft.Colors.WHITE,
            border_radius=12,
        )
    )

    # Inicial
    hoy = datetime.today().date()
    txt_fecha.value = str(hoy)
    page.update()
    predecir_para_fecha(hoy)

if __name__ == "__main__":
    ft.app(target=main)

