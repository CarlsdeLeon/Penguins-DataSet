#Clasificador de Naive Bayes
#pasos: 
#1. Calcular la probabilidad a priori de cada clase
#2. Calcular la probabilidad condicionada de cada atributo dado cada clase
#3. Calcular la aprobabilidad posterior para cada clase dado un nuevo conjunto de atributos
#4. Seleccionar la clase con la mayor probabilidad posterior como la predicción del modelo
 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  
from sklearn import metrics

data = sns.load_dataset('penguins')
data = data.dropna()

train, test = train_test_split(
    data,
    test_size=0.4,
    stratify=data['species'],
    random_state=23,
)

#atributos del dataset
fn = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

#clases
cn = ['Adelie', 'Chinstrap', 'Gentoo']

x_train = train[fn]
y_train = train['species']

#conjunto de pruebas siguiendo el modelo Y = mX + b
x_test = test[fn]
y_test = test['species']

#Clasificador Naive Bayes
mod_gnb = GaussianNB()

mod_gnb.fit(x_train, y_train)

prediccion = mod_gnb.predict(x_test)

print(prediccion)

eficiencia = metrics.accuracy_score(y_test, prediccion)

print("Precisión del modelo:", eficiencia)

test['y_pred'] = prediccion

print(test)

print(metrics.confusion_matrix(y_test, prediccion))

cm = metrics.confusion_matrix(y_test, prediccion)

fig = plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=mod_gnb.classes_,
    yticklabels=mod_gnb.classes_
)

plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Matriz de Confusión')

plt.show()