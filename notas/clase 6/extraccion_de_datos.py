import camelot
archivo = './clase 6/Analisis Anual 2018 ETAS.pdf'
tabla = camelot.read_pdf(archivo, pages='3')
print(tabla)
print(tabla[0].df)
print(tabla[0].df[0][2].split('\n'))
area = list(map(lambda area: area.strip(), tabla[0].df[0][2].split('\n')))
print(area)

