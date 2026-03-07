# CLASIFICADOR K-MEANS (CLUSTERING - NO SUPERVISIONADO)
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d

sns.set() #para ver las fronteras

datos = sns.load_dataset('penguins')

df = sns.load_dataset('penguins')
df = df.dropna()

print(df.info())

print(df.head())

print(df.describe())
print(df.describe().T)
print(df.species.value_counts())

print(df.body_mass_g.hist())
plt.show()

print(df.body_mass_g.plot.hist())
plt.title('Distribucion de masa corporal')
plt.xlabel('Masa en gramos')
plt.show()

# Crear segmentos similares al ejemplo anterior
df['Segmento'] = np.where(df.body_mass_g >=5000, 'Masa Alta',
                          np.where(df.body_mass_g < 4000, 'Masa baja',
                                   'Masa moderada'))

print(df.Segmento.value_counts())
print(df.groupby('Segmento')['body_mass_g'].describe().T)

print(df.plot.scatter(x='flipper_length_mm', y='body_mass_g'))
plt.show()

scaler = StandardScaler()

col_escalar = ['bill_length_mm','bill_depth_mm','flipper_length_mm']

datos_escalados = df.copy()

datos_escalados[col_escalar] = scaler.fit_transform(df[col_escalar])

print(datos_escalados)

print(datos_escalados.plot.scatter(x='flipper_length_mm', y='bill_length_mm'))
plt.show()

from sklearn.cluster import KMeans

modelo = KMeans(n_clusters=3,random_state=16)

modelo.fit(datos_escalados[col_escalar])

print(modelo.labels_)

datos_escalados['Segmento K'] = modelo.predict(datos_escalados[col_escalar])

print(datos_escalados)

datos_escalados['Segmento K'].value_counts()

marcador = ['x', '*', '.']

for segmento in range(3):
    temporal = datos_escalados[datos_escalados['Segmento K']==segmento]
    plt.scatter(temporal.bill_length_mm,
                temporal.bill_depth_mm,
                marker=marcador[segmento],
                label = 'Segmento k'+str(segmento))

datos_escalados[col_escalar].head()

modelo.fit(datos_escalados[col_escalar])

plt.legend()
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

print(ax.scatter3D(datos_escalados['bill_length_mm'],
                   datos_escalados['flipper_length_mm'],
                   datos_escalados['bill_depth_mm'],
                   c=datos_escalados['Segmento'],
                   cmap='tab10'))

ax.set_title('Segmentación de pingüinos')
ax.set_xlabel('Bill length')
ax.set_ylabel('Flipper length')
ax.set_zlabel('Bill depth')

plt.show()

codificador = LabelEncoder()

datos_escalados['Segmento K'] = codificador.fit_transform(datos_escalados['Segmento K'])

print(datos_escalados.head())

fog = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')

print(ax.scatter3D(datos_escalados['bill_length_mm'],
                   datos_escalados['flipper_length_mm'],
                   datos_escalados['bill_depth_mm'],
                   c=datos_escalados['Segmento K'],
                   cmap='tab10'))

ax.set_title('Segmentación de pingüinos')
ax.set_xlabel('Bill length')
ax.set_ylabel('Flipper length')
ax.set_zlabel('Bill depth')

plt.show()