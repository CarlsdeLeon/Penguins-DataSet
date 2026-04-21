#Importar paquetes
from matplotlib import pyplot as plt
import numpy as  np
import imageio
from skimage import color
#Importar OpenCV
import cv2
import common
import pylab

#Modificar las dimensiones de visualizacion

pylab.rcParams['figure.figsize'] = (6.4, 4.0)

ruta = './clase 13/wall-e.jpg'
input_image = cv2.imread(ruta)
plt.imshow(input_image)
plt.show()
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.show()

#inversion vertical

flipped_image = cv2.flip(input_image, 0)
plt.imshow(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
plt.show()

flip_vertical = cv2.flip(input_image, 0)
plt.imshow(cv2.cvtColor(flip_vertical, cv2.COLOR_BGR2RGB))
plt.show()

flip_horizontal = cv2.flip(input_image, 1)
plt.imshow(cv2.cvtColor(flip_horizontal, cv2.COLOR_BGR2RGB))
plt.show()

transpuesta = cv2.transpose(input_image)
plt.imshow(cv2.cvtColor(transpuesta,cv2.COLOR_BGR2RGB))
plt.show()

rotacion90 = cv2.flip(cv2.transpose(input_image),0)
plt.imshow(cv2.cvtColor(rotacion90,cv2.COLOR_BGR2RGB))
plt.show()

rotacion90_1 = cv2.transpose(cv2.flip(input_image,0))
plt.imshow(cv2.cvtColor(rotacion90_1,cv2.COLOR_BGR2RGB))
plt.show()

# Operaciones aritmeticas con imagenes

#crear una matriz h, w, 3
blank_image = np.zeros((input_image.shape), np.uint8)
#generar el marco verde

blank_image[125:255,350:450,1] = 50
plt.imshow(blank_image)
plt.show()

imagen_resaltada = cv2.cvtColor(cv2.add(blank_image, input_image), cv2.COLOR_BGR2RGB)
plt.imshow(imagen_resaltada)
plt.show()

blank_image[125:255,350:450,0] = 50
plt.imshow(blank_image)
plt.show()

imagen_resaltada = cv2.cvtColor(cv2.add(blank_image, input_image), cv2.COLOR_BGR2RGB)
plt.imshow(imagen_resaltada)
plt.show()

blank_image[125:255,350:450,2] = 50
plt.imshow(blank_image)
plt.show()

imagen_resaltada = cv2.cvtColor(cv2.add(blank_image, input_image), cv2.COLOR_BGR2RGB)
plt.imshow(imagen_resaltada)
plt.show()

#filtro de desenfoque gaussiano
d=10
img_desenfoque = cv2.GaussianBlur(input_image,(2 * d + 1, 2 * d + 1), -1)
plt.imshow(cv2.cvtColor(img_desenfoque, cv2.COLOR_BGR2RGB))
plt.show()

#escala de grises
img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray')
plt.show()

sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=9)
plt.imshow(sobelX, cmap='gray')
plt.show()


sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=9)
plt.imshow(sobelY, cmap='gray')
plt.show()


sobelXY = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=9)
plt.imshow(sobelXY, cmap='gray')
plt.show()

# Escala de grises de massy
img_gray_m = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray_m, cmap='gray')
plt.show()

sobelx_m = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=9)
plt.imshow(sobelx_m, cmap='gray')
plt.show()

sobely_m = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=9)
plt.imshow(sobely_m, cmap='gray')
plt.show()

sobelxy_m = cv2.Sobel(input_image, cv2.CV_64F, 1, 1, ksize=9)
plt.imshow(sobelxy_m, cmap='gray')
plt.show()