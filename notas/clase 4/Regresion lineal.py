import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm 
from scipy import stats
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')
print(data.head(10))

print(data['species'].unique())
#data.spices.unique()

print(data['species'].value_counts()) #los grupos qu ehay dentro del conjunto y cuantos elementos hay)

x = data['petal_length']
y = data['petal_width']

# agregar columna constante = 1 para calcular intecepto

x = sm.add_constant(x)
print(x.head())

resultado = sm.OLS(y, x).fit() #OLS = ordinary least squares
print(resultado.summary()) #en la segunda tabla encontraras los valores
#pendiente y = mx + b y + 0.4158 x - 0.3661
#petal_width = 0.4158 * petal_length - 0.3661
#r2 = 0.927 es el indice de determinacion es la primera tabla, no mame :,v
#r = 0.9628 es el indice de correlacion

print(resultado.params) #parametros de la regresion

print(resultado.rsquared) #indice de determinacion

print(np.sqrt(resultado.rsquared)) #resultado de correlacion

#grafico de regrecion

sns.set_theme(color_codes=True)
ax = sns.regplot(data = data, x = 'petal_width', y = 'petal_length') #regration plot
ax.set_xlabel('Largo del petalo')
ax.set_ylabel('Ancho del petalo')
ax.set_title('Regresion lineal entre largo y ancho del petalo')
ax.plot()
plt.show()

ax2 = sns.jointplot(data = data, x = 'petal_width', y = 'petal_length', kind = 'reg', truncate = False, color='m', height=7)
plt.show()

sns.pairplot(data)
plt.show()
