import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importar el separador de muestras
from sklearn.model_selection import train_test_split

# Importar el clasificador
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Importar metricas
from sklearn import metrics


data = sns.load_dataset('penguins')

# eliminar valores nulos
data = data.dropna()

# dividir entrenamiento y prueba
train, test = train_test_split(
    data,
    test_size=0.4,
    stratify=data['species'],
    random_state=42
)

print(train)


fn = ['bill_length_mm', 'bill_depth_mm',
      'flipper_length_mm', 'body_mass_g']


cn = ['Adelie', 'Chinstrap', 'Gentoo']

x_train = train[fn]
y_train = train['species']

x_test = test[fn]
y_test = test['species']


mod_dt = DecisionTreeClassifier(max_depth=3, random_state=46)

mod_dt.fit(x_train, y_train)

prediccion = mod_dt.predict(x_test)

print(prediccion)

print("Importancia de características:")
print(mod_dt.feature_importances_)


plt.figure(figsize=(12, 8))

plot_tree(
    mod_dt,
    feature_names=fn,
    class_names=cn,
    filled=True
)

plt.show()


eficiencia = metrics.accuracy_score(y_test, prediccion)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("Precisión del modelo:", eficiencia)

error = pd.DataFrame(x_test)
error['species_real'] = y_test
error['prediccion'] = prediccion

print(error)
