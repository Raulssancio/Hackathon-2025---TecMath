import pandas as pd
import numpy as np

# Load your CSV (replace path with your actual file)
df = pd.read_csv("C:\\Users\\octab\\Downloads\\Datos Hackaton 2025\\Temperatura del aire cerca del suelo.csv")

# Convert 'FechaHora' to datetime
df['FechaHora'] = pd.to_datetime(df['FechaHora'])

# Extract components
df['Year'] = df['FechaHora'].dt.year
df['Month'] = df['FechaHora'].dt.month
df['Day'] = df['FechaHora'].dt.day
df['Hour'] = df['FechaHora'].dt.hour

# Resulting DataFrame contains separated columns
#print(df.head())


mes = int(input("mes: "))
dia = int(input("dia: "))
hora = int(input("hora: "))

# Assuming previous steps have been done and df has 'year', 'month', 'day' columns
filtered_month_8 = df[(df['Month'] == mes) & (df['Day'] == dia) & ((df['Hour'] == hora) | (df['Hour'] == hora+1))]

# Display only the temperatures from this selection
y = filtered_month_8['Temp'].values
print(y)
tamano = len(y)
x = []

for i in range(tamano):
    x.append(i)

def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)

    n = len(x)
    m = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x**2) - (np.sum(x))**2)
    b = (np.sum(y) - m*np.sum(x)) / n
    return m, b

m, b = linear_regression(x, y)
print(m)
print(b)

annnno = int(input("anio: "))
anio = annnno - 2015
aprox = m * anio + b
print(aprox)