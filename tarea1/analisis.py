import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates

datos = sns.load_dataset('penguins')

datos = datos.dropna()

print(datos)
print(datos.describe())

print(datos['species'].size)

print(datos.groupby('species').size())

train, test = train_test_split(
    datos,
    test_size=0.4,
    stratify=datos['species'],
    random_state=42
)

print(train)


n_bins = 10
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

ax1.hist(train['bill_length_mm'], bins=n_bins)
ax1.set_title('Longitud del pico (mm)')

ax2.hist(train['bill_depth_mm'], bins=n_bins)
ax2.set_title('Profundidad del pico (mm)')

ax3.hist(train['flipper_length_mm'], bins=n_bins)
ax3.set_title('Longitud de aleta (mm)')

ax4.hist(train['body_mass_g'], bins=n_bins)
ax4.set_title('Masa corporal (g)')

fig.tight_layout(pad=1.0)
plt.show()


fig1, axs1 = plt.subplots(nrows=2, ncols=2)

fn = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']

sns.boxplot(x='species', y=fn[0], data=train, ax=axs1[0, 0])
sns.boxplot(x='species', y=fn[1], data=train, ax=axs1[0, 1])
sns.boxplot(x='species', y=fn[2], data=train, ax=axs1[1, 0])
sns.boxplot(x='species', y=fn[3], data=train, ax=axs1[1, 1])

fig1.tight_layout(pad=1.0)
plt.show()


sns.set_palette("Set2")

sns.violinplot(
    x='species',
    y='bill_length_mm',
    data=train,
    hue='species',
    palette='colorblind'
)

plt.show()



sns.pairplot(data=train, hue='species', height=2, palette='colorblind')
plt.show()



corrmat = train.corr(numeric_only=True)
print(corrmat)

sns.heatmap(corrmat, annot=True, square=True)
plt.show()


datos_parallel = train[['bill_length_mm','bill_depth_mm',
                        'flipper_length_mm','body_mass_g','species']]

parallel_coordinates(datos_parallel, 'species',
                     color=['red', 'green', 'blue'])

plt.show()
