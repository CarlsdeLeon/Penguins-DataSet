#matrices de peso y funcion de activacion
import math
import numpy as np
import seaborn as sns
W1 = np.array([[9.37285205001324,-4.66766267972539],
[0.882311706625367,0.0202416015144118],
[1.21399452658106,-2.46092474628536],
[-2.69242365090413,2.64840298050765],
[-3.86639116449596,0.231707555193143]])

W2 = np.array([[0.979486230422685,2.39868091700382,-0.276685237716272],
[245.485299995854,3.00758809493884,0.824766525197065],
[-66.6721214283505,1.28328541603641,-69.7696444862307]])

W3 = np.array([[-0.409723143375303,0.0648935512829711,0.0377039512117264],
[-0.00800212119510466,1.00356087705685,-1.01986435328965],
[0.419568883290797,-0.067248468369046,0.986119764097346],
[1.58515766818101,-1.58755727229418,0.000754949849812554]])

def f_act(X):
    activada = np.array([1/(1+ np.exp(-x)) for x in X], dtype=np.float64)
    return activada

iris = sns.load_dataset('iris')

print(iris.head())
print(iris.info())

xcols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
especie = iris['species'].unique()

print(especie)

X = iris[xcols].copy()

X.insert(0, 'bias', 1)

print(X)

#prediccion

prediccion = []

for index, fila in X.iterrows():
    capa1 = f_act(fila.dot(W1))
    capa1 = np.insert(capa1, 0, 1)

    capa2 = f_act(capa1.dot(W2))
    capa2 = np.insert(capa2, 0, 1)

    salida = f_act(capa2.dot(W3))

    prediccion.append(especie[np.argmax(salida)])
iris['Prediccion'] = prediccion
print(iris.head())

errorneas = iris[iris['species'] != iris['Prediccion']]
print(errorneas)

eficiencia = (1- len(errorneas)/len(iris))*100
print(eficiencia)