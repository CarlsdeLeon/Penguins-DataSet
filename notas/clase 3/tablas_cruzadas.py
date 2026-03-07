import pandas as pd #para manejo de datos
import seaborn as sns #para visualizacion de datos

ruta_archivo = 'datos.xlsx'

tabla = pd.read_excel(ruta_archivo)#cargar dataset de vuelos

contingencia = pd.crosstab(tabla['Edad'], tabla['Sexo'])
print('tabla de contingenci - species vs sex:')
print(contingencia)

# calculo intervalos de edad
n = len(tabla)
intervalos = pd.cut(tabla['Edad'], bins=5)
print(intervalos)

#calcular frecuencias por intervalos   
frecuencias = pd.crosstab(index=intervalos, columns='f')

frecuencias['fa'] = frecuencias['f'].cumsum() #frecuencia acumulada
frecuencias['fr'] = frecuencias['f'] / n #frecuencia relativa
frecuencias['fra'] = frecuencias['fr'].cumsum() #frecuencia relativa acumulada
frecuencias['%'] = frecuencias['fr'] * 100 #frecuencia porcentual
frecuencias['%a'] = frecuencias['fra'] * 100 #frecuencia porcentual acumulada
print(frecuencias)

print(type(intervalos[0]))
print(intervalos[0].left) #limite inferior del intervalo
print(frecuencias.index.categories.mid)

frecuencias['Xm']=frecuencias.index.categories.mid #punto medio del intervalo
print(frecuencias)
frecuencias['f*Xm']=frecuencias['f']*frecuencias['Xm']
print(frecuencias)
frecuencias['f*Xm'].sum()/n #calculo de la media
