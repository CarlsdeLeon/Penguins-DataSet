#Modulos requeridos
import numpy as np #para manipulacion
import pandas as pd #para manejo de datos
import seaborn as sns #para visualizacion de datos
import matplotlib.pyplot as plt #para graficacion
import yfinance as yf #para datos financieros

vuelos = sns.load_dataset('flights') #cargar dataset de vuelos
print(vuelos.head()) #mostrar las primeras filas del dataset

ruta_archivo = './clase 2/contenido/datos.xlsx' #ruta del archivo CSV

tabla = pd.read_excel(ruta_archivo)
print(tabla)
#leer archivpo CSV
planilla = pd.read_csv('./clase 2/contenido/planilla.csv')
planilla.set_index('codigo', inplace=True)
planilla.dtypes
planilla['fecha'] = pd.to_datetime(planilla['fecha_contratacion'])
planilla.dtypes

#yahoo finances
ticker_apple = 'AAPL'
start_date = '2026-01-01'
end_date = '2026-01-19'
df_apple = yf.download(ticker_apple, start=start_date, end=end_date)
print(df_apple.head())

