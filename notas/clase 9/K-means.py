# CLASIFICADOR K-MEANS (CLUSTERING - NO SUPERVISIONADO)
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d

sns.set() #para ver las fronteras

ruta = 'clase 9/Mall_Customers-2.csv'
df = pd.read_csv(ruta, index_col=0)
print(df.info())

df.rename({'Gender':'Genero', 'Age':'Edad','Annual Income (k$)':'Ingreso','Spending Score (1-100)':'Gasto'}, axis=1, inplace=True)

print(df.head())

print(df.describe())
print(df.describe().T)
print(df.Genero.value_counts())
print(df.Ingreso.hist())
plt.show()

print(df.Ingreso.plot.hist())
plt.title('Distribucion de ingreso Anual')
plt.xlabel('Ingreso en Miles de USD'); 
plt.show()

df['Segmento'] = np.where(df.Ingreso >=90, 'Ingreso Alto',
                          np.where(df.Ingreso < 50, 'Ingreso bajo',
                                   'Ingreso moderado'))

print(df.Segmento.value_counts())
print(df.groupby('Segmento')['Ingreso'].describe().T)
print(df.plot.scatter(x='Ingreso', y='Gasto'))
plt.show()

scaler = StandardScaler()
col_escalar = ['Edad', 'Ingreso', 'Gasto']
datos_escalados = df.copy()
datos_escalados[col_escalar] = scaler.fit_transform(df[col_escalar])

print(datos_escalados)
print(datos_escalados.plot.scatter(x='Ingreso', y='Gasto'))
plt.show()

from sklearn.cluster import KMeans

modelo = KMeans(n_clusters=5,random_state=16)
modelo.fit(datos_escalados[col_escalar])
print(modelo.labels_)

datos_escalados['Segmento K'] = modelo.predict(datos_escalados[col_escalar])
print(datos_escalados)

datos_escalados['Segmento K'].value_counts()

marcador = ['x', '*', '.', '|','_']

for segmento in range(5):
    temporal = datos_escalados[datos_escalados['Segmento K']==segmento]
    plt.scatter(temporal.Ingreso, temporal.Gasto, marker=marcador[segmento], #no funciona arregla los marcadores
                label = 'Segmento k'+str(segmento))

datos_escalados[col_escalar].head()
modelo.fit(datos_escalados[col_escalar])

plt.show()

print(datos_escalados.head())
print(datos_escalados['Segmento'].value_counts())

print(datos_escalados.head())

from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()
datos_escalados['Segmento'] = codificador.fit_transform(datos_escalados['Segmento'])
print(datos_escalados.head())

fog = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
print(ax.scatter3D(datos_escalados['Edad'], datos_escalados['Ingreso'], datos_escalados['Gasto'], c=datos_escalados['Segmento'], cmap='tab10')); 
ax.set_title('Segmentación de clientes')
ax.set_xlabel('Edad')
ax.set_ylabel('Ingreso')
ax.set_zlabel('Gasto')

plt.show()

codificador = LabelEncoder()
datos_escalados['Segmento K'] = codificador.fit_transform(datos_escalados['Segmento K'])
print(datos_escalados.head())

fog = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
print(ax.scatter3D(datos_escalados['Edad'], datos_escalados['Ingreso'], datos_escalados['Gasto'], c=datos_escalados['Segmento K'], cmap='tab10')); 
ax.set_title('Segmentación de clientes')
ax.set_xlabel('Edad')
ax.set_ylabel('Ingreso')
ax.set_zlabel('Gasto')

plt.show()

