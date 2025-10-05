import tkinter as tk
from tkcalendar import DateEntry
from datetime import datetime
import pandas as pd
from PIL import Image, ImageTk, ImageSequence

# ---------------------- Config ----------------------
COLOR_FONDO = "#f0f4f8"
COLOR_BOTON = "#4CAF50"
COLOR_TEXTO = "#333333"
FUENTE_TITULO = ("Segoe UI", 16, "bold")
FUENTE_TEXTO = ("Segoe UI", 12)

# ---------------------- Función para procesar temperatura ----------------------
def generar_csv_diario(temp_csv):
    temp_df = pd.read_csv(temp_csv)
    temp_df.columns = temp_df.columns.str.strip()
    temp_df['FechaHora'] = pd.to_datetime(temp_df['FechaHora'])
    temp_df['Temp'] = temp_df['Temp'] - 273.15  # Kelvin → Celsius
    temp_df['Fecha'] = temp_df['FechaHora'].dt.date
    df_diario = temp_df.groupby('Fecha')['Temp'].mean().reset_index()
    return df_diario

df_diario = generar_csv_diario("Temperatura del aire cerca del suelo.csv")

# ---------------------- Etiquetas y GIFs ----------------------
def obtener_etiqueta(temp):
    if temp < 10:
        return "Frío", "frio.gif"
    elif temp < 20:
        return "Templado", "templado.gif"
    elif temp < 30:
        return "Caluroso", "caluroso.gif"
    else:
        return "Muy Caluroso", "muy_caluroso.gif"

# ---------------------- Animar GIF ----------------------
def animar_gif(label, gif_path, resize=(250, 250), delay=20):
    """
    Animar un GIF en un Label de Tkinter.
    
    label   : Label donde se mostrará el GIF
    gif_path: ruta del GIF
    resize  : tupla (ancho, alto) para redimensionar el GIF
    delay   : milisegundos entre frames (50 = 20 fps)
    """
    # Abrir GIF
    gif = Image.open(gif_path)
    
    # Extraer frames
    frames = []
    try:
        while True:
            frame = gif.copy().convert("RGBA")
            frame = frame.resize(resize, Image.Resampling.LANCZOS)
            frames.append(ImageTk.PhotoImage(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass  # fin del GIF

    # Loop de animación
    def loop(idx=0):
        label.config(image=frames[idx])
        label.image = frames[idx]  # evitar garbage collection
        label.after(delay, loop, (idx + 1) % len(frames))

    loop()  # iniciar animación

# ---------------------- Función Tkinter ----------------------
def abrir_prediccion():
    pred_win = tk.Toplevel(root)
    pred_win.title("Predicción de Temperatura")
    pred_win.geometry("450x500")
    pred_win.configure(bg=COLOR_FONDO)

    tk.Label(pred_win, text="Predicción de temperatura", font=FUENTE_TITULO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=15)

    tk.Label(pred_win, text="Selecciona la fecha:", font=FUENTE_TEXTO,
             bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=10)
    
    fecha_entry = DateEntry(pred_win, width=12, background='darkblue',
                            foreground='white', borderwidth=2, font=FUENTE_TEXTO)
    fecha_entry.pack(pady=5)

    resultado = tk.Label(pred_win, text="", font=FUENTE_TEXTO, bg=COLOR_FONDO)
    resultado.pack(pady=10)

    gif_label = tk.Label(pred_win, bg=COLOR_FONDO)
    gif_label.pack(pady=10)

    def predecir():
        fecha = fecha_entry.get_date()
        fila = df_diario[df_diario['Fecha'] == fecha]
        if not fila.empty:
            temp = fila.iloc[0]['Temp']
            etiqueta, gif_file = obtener_etiqueta(temp)
            resultado.config(text=f"Temp promedio el {fecha}: {temp:.1f}°C ({etiqueta})")
            animar_gif(gif_label, gif_file)
        else:
            resultado.config(text="No hay datos para esa fecha.")
            gif_label.config(image="")

    tk.Button(pred_win, text="Mostrar Temperatura", command=predecir,
              font=FUENTE_TEXTO, bg=COLOR_BOTON, fg="white", relief="flat").pack(pady=10)

# ---------------------- Ventana Principal ----------------------
root = tk.Tk()
root.title("App del Clima - AIQ")
root.geometry("400x220")
root.configure(bg=COLOR_FONDO)

tk.Label(root, text="☁️ App del Clima AIQ ☁️", font=FUENTE_TITULO,
         bg=COLOR_FONDO, fg=COLOR_TEXTO).pack(pady=20)

tk.Button(root, text="Predicción de Temperatura", command=abrir_prediccion,
          font=FUENTE_TEXTO, bg=COLOR_BOTON, fg="white",
          relief="flat", padx=10, pady=5, width=25).pack(pady=20)

root.mainloop()
