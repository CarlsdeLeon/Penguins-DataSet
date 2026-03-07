import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importar el separador de muestras para entrenamiento y pruebas
from sklearn.model_selection import train_test_split
#Importar el clasificador de arbol de decision
from sklearn.tree import DecisionTreeClassifier, plot_tree
#Importar las mericas para medir la eficiencia del modelo
from sklearn import metrics

#cargar conjunto de datos
data = sns.load_dataset('iris')
#seleccionar datos de prueba y entrenamiento
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
print(train)

fn = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
cn = ['setosa', 'versicolor', 'virginica']

x_train = train[fn]
y_train = train['species']

#conjunto de pruebas siguiendo el modelo Y = mX + b
x_test = test[fn]
y_test = test['species']

#Arbol de decision
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=112)
#ajustar el modelo
mod_dt.fit(x_train, y_train)
#probar el modelo 
prediccion = mod_dt.predict(x_test)

print(prediccion)

#vemos que es mas importante para el modelo
print(mod_dt.feature_importances_)

#visualizar las reglas de clasificacion
plt.figure(figsize=(10, 8))

plot_tree(mod_dt, feature_names=fn, class_names=cn, filled=True); 

plt.show()

#Eficiencia
eficiencia = metrics.accuracy_score(y_test, prediccion)
print(eficiencia)

error = pd.DataFrame(x_test)
error['species'] = y_test
error['prediccion'] = prediccion
print(error)

