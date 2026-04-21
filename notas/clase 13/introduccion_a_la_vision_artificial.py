from matplotlib import pyplot as plt
import numpy as  np
import imageio
from skimage import color

#Lectura de la imagen
ruta = "./clase 13/giorno-jojo-art2.jpg"
imgIn = imageio.v2.imread(ruta)

#Caracteristicas de la imagen = matriz w * h * 3
print('Dimesiones de la imagen:', imgIn.shape)
print('Tipos de dato:', imgIn.dtype)

#Imprimir el contenido de las capas de color
print('Print en [0,0,0]', imgIn[0,0,0])
print('Print en [0,0,1]', imgIn[0,0,1])
print('Print en [0,0,2]', imgIn[0,0,2])
print('Print en [0,0,:]', imgIn[0,0,:])

plt.imshow(imgIn)
plt.show()
plt.imshow(imgIn[:,:,1])
plt.show()
imgGray = color.rgb2gray(imgIn)
plt.imshow(imgGray, cmap='gray')
plt.show()
print('Dimensiones de la imagem:', imgGray.shape)
print('Tipo de datos:', imgGray.dtype)
imgSeccion = imgGray[10:250,35:250]
plt.imshow(imgSeccion, cmap='gray')
plt.show()

imgModificada = imgGray.copy()
imgModificada[imgModificada < 0.2] = 0
plt.imshow(imgModificada, cmap='gray')
plt.show()

plt.hist(imgGray.flatten(),bins=100)
plt.show()

imageio.imwrite('clase 13/giorno-jojo-art2-modificado.jpg', (imgModificada * 255).astype(('uint8')))

