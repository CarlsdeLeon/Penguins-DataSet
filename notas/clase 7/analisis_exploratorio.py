import pandas as pd
import numpy
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#cargar conjunto de datos
datos = sns.load_dataset('iris')
print(datos)

print(datos.describe())

print(datos['species'].size)

#aqui podemos ver cuantas aparecen de cada clase de flor
print(datos.groupby('species').size())

#seleccionar datos de prueba y entrenamiento
train, test = train_test_split(datos, test_size=0.4, stratify=datos['species'], random_state=42)
print(train)
      
#expliracion visual

#generacion de histogramas
# Exploracion visual
# Generacion de histogramas
n_bins = 10
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.hist(train['sepal_length'], bins=n_bins)
ax1.set_title('Longitud de sepalo')
ax2.hist(train['sepal_width'], bins=n_bins)
ax2.set_title('Ancho de sepalo')
ax3.hist(train['sepal_length'], bins=n_bins)
ax3.set_title('Longitud de sepalo')
ax4.hist(train['sepal_width'], bins=n_bins)
ax4.set_title('Ancho de sepalo')
fig.tight_layout(pad=1.0)
plt.show()


#generacion de diagramas de caja
fig1,axs1 = plt.subplots(nrows = 2, ncols=2)
fn = ['sepal_length','sepal_width','petal_length','petal_width']
cn = ['setosa','versicolor','virginica']
sns.boxplot(x='species',y=fn[0],data=train,ax=axs1[0, 0])
sns.boxplot(x='species',y=fn[1],data=train,ax=axs1[0, 1])
sns.boxplot(x='species',y=fn[2],data=train,ax=axs1[1, 0])
sns.boxplot(x='species',y=fn[3],data=train,ax=axs1[1, 1])

fig1.tight_layout(pad=1.0)

plt.show()

sns.set_palette("Set2")

#grafivo de violin
sns.violinplot(x='species',y='sepal_length',data=train, hue='species', order=cn, palette='colorblind'); 

plt.show()

#diagrana de dispersion de atributos emparejados
sns.pairplot(data=train, hue='species', height=2, palette='colorblind')
plt.show()  

#Matriz de correlacion
corrmat = train.corr(numeric_only=True)
print(corrmat)

#Mapa de calor de la matriz de correlacion
sns.color_palette("mako", as_cmap=True)
sns.heatmap(corrmat, annot=True, square=True); 

plt.show()

# Coordenadas paralelas
from pandas.plotting import parallel_coordinates
parallel_coordinates(train, 'species', color=['red', 'green', 'blue'])
plt.show()




