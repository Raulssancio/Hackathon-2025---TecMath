import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from datetime import datetime
import pandas as pd
from PIL import Image, ImageTk

# ---------------------- Configuración ----------------------
COLOR_FONDO = "#f0f4f8"
COLOR_BOTON = "#4CAF50"
COLOR_BOTON_SEC = "#2196F3"
COLOR_TEXTO = "#333333"
FUENTE_TITULO = ("Segoe UI", 16, "bold")
FUENTE_TEXTO = ("Segoe UI", 12)

# Lista de ciudades (limitada al AIQ)
ciudades = ["AIQ - Aeropuerto de Querétaro"]

# ---------------------- Función para generar CSV diario ----------------------
def generar_csv_diario(temp_csv):
    temp_df = pd.read_csv(temp_csv)
    temp_df.rename(columns={'FechaHora': 'FechaHora', 'Temp': 'Temp'}, inplace=True)

    # Convertir a datetime
    temp_df['FechaHora'] = pd.to_datetime(temp_df['FechaHora'])

    # Ajuste de 12 horas (corrige 3:00 -> 15:00)
    temp_df['FechaHora'] = temp_df['FechaHora'] + pd.Timedelta(hours=12)

    # Conversión Kelvin -> Celsius
    temp_df['Temp'] = temp_df['Temp'] - 273.15

    return temp_df

# ---------------------- Cargar CSV ----------------------
df_temp = generar_csv_diario("Temperatura del aire cerca del suelo.csv")

# ---------------------- Función para animar GIF ----------------------
def animar_gif(label, frames, delay, index=0):
    frame = frames[index]
    label.configure(image=frame)
    label.image = frame
    label.after(delay, animar_gif, label, frames, delay, (index + 1) % len(frames))

# ---------------------- Funciones Tkinter ----------------------
def abrir_temperaturas_dia():
    temp_win = tk.Toplevel(root)
    temp_win.title("Temperaturas del Día")
    temp_win.geometry("520x500")
    temp_win.configure(bg=COLOR_FONDO)

    tk.Label(temp_win, text="Temperaturas del día", font=FUENTE_TITULO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=15)

    tk.Label(temp_win, text="Ciudad: AIQ - Aeropuerto de Querétaro", font=FUENTE_TEXTO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=5)

    tk.Label(temp_win, text="Selecciona la fecha:", font=FUENTE_TEXTO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=10)

    fecha_entry = DateEntry(temp_win, width=12, background='darkblue',
                            foreground='white', borderwidth=2,
                            font=FUENTE_TEXTO)
    fecha_entry.pack(pady=5)

    resultado = tk.Label(temp_win, text="", font=FUENTE_TEXTO, bg=COLOR_FONDO, justify="left")
    resultado.pack(pady=10)

    gif_label = tk.Label(temp_win, bg=COLOR_FONDO)
    gif_label.pack(pady=10)

    def mostrar_temperaturas():
        fecha_sel = fecha_entry.get_date()
        df_dia = df_temp[df_temp['FechaHora'].dt.date == fecha_sel]

        if df_dia.empty:
            resultado.config(text="No hay datos para esa fecha.")
            gif_label.config(image="")
            return

        # Agrupar cada 3 horas
        df_dia_resample = df_dia.set_index('FechaHora').resample('3H').mean()

        # Construir texto con temperaturas
        texto = f"🌡️ Temperaturas para el {fecha_sel}:\n\n"
        for idx, row in df_dia_resample.iterrows():
            texto += f"{idx.strftime('%H:%M')}: {row['Temp']:.1f} °C\n"

        temp_max = df_dia['Temp'].max()
        temp_min = df_dia['Temp'].min()
        temp_prom = df_dia['Temp'].mean()

        texto += f"\n🔥 Máxima: {temp_max:.1f} °C"
        texto += f"\n❄️ Mínima: {temp_min:.1f} °C"
        texto += f"\n📊 Promedio: {temp_prom:.1f} °C"

        # Determinar etiqueta y gif
        if temp_prom < 10:
            etiqueta = "Frío 🧊"
            gif_path = "frio.gif"
        elif temp_prom < 20:
            etiqueta = "Templado 🌤️"
            gif_path = "templado.gif"
        elif temp_prom < 30:
            etiqueta = "Caluroso 🔥"
            gif_path = "caluroso.gif"
        else:
            etiqueta = "Muy caluroso ☀️"
            gif_path = "muy_caluroso.gif"

        texto += f"\n\nEstado del día: {etiqueta}"
        resultado.config(text=texto)

        # Mostrar GIF animado
        try:
            gif_image = Image.open(gif_path)
            frames = []
            try:
                while True:
                    frame = gif_image.copy()
                    frame.thumbnail((250, 250))
                    frames.append(ImageTk.PhotoImage(frame))
                    gif_image.seek(len(frames))
            except EOFError:
                pass
            animar_gif(gif_label, frames, 25)
        except Exception as e:
            gif_label.config(text="(No se pudo cargar el GIF)", image="")

    tk.Button(temp_win, text="Mostrar temperaturas", command=mostrar_temperaturas,
              font=FUENTE_TEXTO, bg=COLOR_BOTON, fg="white", relief="flat").pack(pady=15)

# ---------------------- Ventana Principal ----------------------
root = tk.Tk()
root.title("App del Clima - AIQ")
root.geometry("420x280")
root.configure(bg=COLOR_FONDO)

titulo = tk.Label(root, text="☁️ App del Clima AIQ ☁️", font=FUENTE_TITULO,
                  bg=COLOR_FONDO, fg=COLOR_TEXTO)
titulo.pack(pady=20)

btn_temp_dia = tk.Button(root, text="Ver temperaturas del día", command=abrir_temperaturas_dia,
                         font=FUENTE_TEXTO, bg=COLOR_BOTON_SEC, fg="white",
                         relief="flat", padx=10, pady=5, width=25)
btn_temp_dia.pack(pady=10)

root.mainloop()


