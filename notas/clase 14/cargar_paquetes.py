import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('./clase 14/ej02-auto.jpg')
img_copy = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
grad = np.absolute(grad)
grad = cv2.convertScaleAbs(grad)

plt.imshow(grad, cmap='gray')
plt.title("Gradiente")
plt.show()

blur = cv2.GaussianBlur(grad, (5,5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.imshow(thresh, cmap='gray')
plt.title("Threshold")
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

plt.imshow(morph, cmap='gray')
plt.title("Morph")
plt.show()

contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# ordenar y limitar
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    area = w * h
    ratio = w / float(h) if h != 0 else 0

    # 🔥 FILTRO SIMPLE (solo forma)
    if area > 300 and 1 < ratio < 8:

        placa = gray[y:y+h, x:x+w]

        # 🔥 1. agrandar más
        placa = cv2.resize(placa, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # 🔥 2. mejorar contraste
        placa = cv2.equalizeHist(placa)

        # 🔥 3. blur suave (quita ruido)
        placa = cv2.GaussianBlur(placa, (3,3), 0)

        # 🔥 4. threshold más agresivo
        _, placa = cv2.threshold(placa, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 🔥 5. MORFOLOGÍA (CLAVE)
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        placa = cv2.morphologyEx(placa, cv2.MORPH_CLOSE, kernel_text)

        config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        texto = pytesseract.image_to_string(placa, config=config)

        # 🔥 limpiar texto
        texto = texto.strip().replace(" ", "").replace("\n", "")

        longitud = len(texto)

        # 🔥 VALIDACIÓN SIMPLE
        if 6 <= longitud <= 8:

            y_text = y - 10 if y - 10 > 10 else y + 30

            cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,255,0), 3)

            cv2.putText(
                img_copy,
                texto,
                (x, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2,
                cv2.LINE_AA
            )

            plt.imshow(placa, cmap='gray')
            plt.title("Placa candidata")
            plt.show()

            print("Texto:", texto)

plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.title("Resultado")
plt.axis('off')
plt.show()