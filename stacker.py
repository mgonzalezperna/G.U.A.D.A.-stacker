import the modules
import time
import os
import PIL
import numpy as np
from PIL import Image
from typing import Tuple
from numpy import ndarray
from itertools import chain


class Offset:
    # Los valores son sobre un eje X o Y
    value = 0  # Valor en cantidad de pixeles que se movio la imagen actual
    # Ademas se utiliza para ignorar los datos de los bordes en la alineacion
    # (Se podria separar en derecha/izquierda o arriba/abajo)
    diff = float('inf')  # Diferencia entre la referencia y la actual
    reference: ndarray  # Imagen de referencia


class ImgOffsets:
    # Define los valores para los dos ejes
    x: Offset
    y: Offset

    def __init__(self):
        self.x = Offset()
        self.y = Offset()

    def getTotalDiff(self) -> float:
        return self.x.diff + self.y.diff


def stack(im: ndarray) -> ndarray:
    # Apilo la imagen con el resultado previo
    offset = findAlignment(im)  # Busco la mejor alineacion

    # Muevo los valores (pixeles) para alinear la imagen
    im = np.roll(im, offset.x.value, axis=1)  # En el eje X (Ancho)
    im = np.roll(im, offset.y.value, axis=0)  # En el eje Y (Alto)

    # Si se movio la imagen una cantidad mayor a las veces previas
    # guardo la cantidad para ignorar el borde en los proximos apilamientos
    cutOffset.x.value = max(cutOffset.x.value, abs(offset.x.value))
    cutOffset.y.value = max(cutOffset.y.value, abs(offset.y.value))

    # Sumo la matriz resultado con la actual que fue alineada
    return np.add(result, im, dtype=np.uint32, out=result)


def getReferenceArray(img: ndarray) -> Tuple[ndarray, ndarray]:
    # Obtengo los valores de los ejes
    img = np.clip(img, 128, None)  # Corto los colores mas oscuros
    # Me interesa los mas claros que es donde estan las estrellas
    # Si no lo hago el ruido en la oscuridad puede afectar
    # demaciado el resultado del alineamiento
    # Sumo los valores de los ejes y me queda una matriz de 2 dimensiones
    arrayx: ndarray = img.sum(axis=0, dtype=np.uint32)  # Ancho x 3 colores
    arrayy: ndarray = img.sum(axis=1, dtype=np.uint32)  # Alto x 3 colores
    return arrayx, arrayy


def findAlignment(im: ndarray) -> ImgOffsets:
    # Busco la mejor alineacion

    # Tomo el promedio de las imagenes de referencia (ya que se van sumando)
    referenceIm1x: ndarray = cutOffset.x.reference // totalStacks
    referenceIm1y: ndarray = cutOffset.y.reference // totalStacks

    # Tomo los valores de referencia para los ejes
    referenceIm2x, referenceIm2y = getReferenceArray(im)

    bestOffset = ImgOffsets()  # Contiene el mejor resultado obtenido

    # Se intenta un alineamiento inicial sin desfasaje
    tryAlignment(referenceIm1x, referenceIm2x,
                 bestOffset.x, 0, cutOffset.x.value)  # Para eje X
    tryAlignment(referenceIm1y, referenceIm2y,
                 bestOffset.y, 0, cutOffset.y.value)  # Para eje Y

    # Se intenta alineamientos con un desfasaje de -9 a 9
    for off in chain(range(-9, 0), range(1, 10)):
        # Para eje X
        tryAlignment(referenceIm1x, referenceIm2x,
                     bestOffset.x, off, cutOffset.x.value)
        # Para eje Y
        tryAlignment(referenceIm1y, referenceIm2y,
                     bestOffset.y, off, cutOffset.y.value)

    print("Aligned at: x:" + str(bestOffset.x.value) + " y:" +
          str(bestOffset.y.value) + " v:" + str(bestOffset.getTotalDiff()))

    # Muevo los valores (pixeles) de las referencias de la imagen con el
    # mejor valor de alineamiento
    referenceIm2x = np.roll(referenceIm2x, bestOffset.x.value, axis=0)
    referenceIm2y = np.roll(referenceIm2y, bestOffset.y.value, axis=0)

    # Agrego los valores de referencia de la imagen al total global
    np.add(cutOffset.x.reference, referenceIm2x,
           dtype=np.uint32, out=cutOffset.x.reference)
    np.add(cutOffset.y.reference, referenceIm2y,
           dtype=np.uint32, out=cutOffset.y.reference)

    return bestOffset


def tryAlignment(im1: ndarray, im2: ndarray, bestOffsetAxis: Offset,
                 offset: int, cutValue: int) -> None:
    # Pruevo el alineamiento con los valores dados
    diff = getAlignment(im1, im2, offset, cutValue)
    if diff < bestOffsetAxis.diff:
        # Si la diferencia es menor a todas las previas guardo su valor
        bestOffsetAxis.diff = diff
        bestOffsetAxis.value = offset


def getAlignment(im1: ndarray, im2: ndarray, offset: int,
                 cutValue: int) -> float:
    # Calculo cuanto de los bordes tengo que cortar
    toCutIm1S: int = max(cutValue, offset)
    toCutIm1E: int = max(cutValue, -offset)
    toCutIm2S: int = max(0, cutValue - offset)
    toCutIm2E: int = max(0, cutValue + offset)

    # Tomo las imagenes cortadas
    im1C: ndarray = im1[toCutIm1S:im1.shape[0] - 1 - toCutIm1E, :]
    im2C: ndarray = im2[toCutIm2S:im2.shape[0] - 1 - toCutIm2E, :]

    imDiff: ndarray = np.subtract(im1C, im2C, dtype=np.int32)  # Diferencia
    imDiff = np.absolute(imDiff, out=imDiff)  # Diferencia absoluta (positiva)
    return imDiff.mean()  # Promedio de diferencia

start_time = time.time()  # Tomo el tiempo de ejecucion para el script
cutOffset = ImgOffsets()  # Guardara el maximo de corrimiento que tuvo una
# imagen y las imagenes de referencia
totalStacks = 1  # Total de imagenes procesadas (Para calcular el promedio)
result: ndarray = None  # Imagen resultado como una matriz de 3 dimensiones

# Busco todas las imagenes en la carpeta l que tengan extension .tif
for file in os.listdir("l"):
    if file.endswith(".tif"):
        filePath = os.path.join("l", file)
        print("Apilamiento " + str(totalStacks) + " con " + filePath)
        start_time_A = time.time()  # Tomo el tiempo proceso para la imagen
        img: Image = Image.open(filePath)  # Abro la imagen
        imArray: ndarray = np.asarray(img)  # Tomo la matriz
        # (Alto x Ancho x 3 colores)
        img.close()
        if result is None:  # La primera imagen la guardo como resultado
            result = np.uint32(imArray)
            cutOffset.x.reference, cutOffset.y.reference = getReferenceArray(
                imArray)  # Guardo los valores de referencia para los ejes
        else:
            result = stack(imArray)  # Inicio el apilamiento de la imagen
            totalStacks += 1
        print("Elapsed time: " + str(time.time() - start_time_A))

# Tomo el promedio (ya que se suman)
finalResult: ndarray = np.uint8(result // totalStacks)

# finalResult = np.clip(finalResult, 8, None)-8

image: Image = Image.fromarray(finalResult)  # Convierto la matriz en imagen

# image = image.filter(
#             ImageFilter.UnsharpMask(radius=2, percent=250, threshold=7))

image.save(time.strftime("%Y%m%d-%H%M%S") + '.tif', 'TIFF')  # La guardo
image.close()

print("Total elapsed time: " + str(time.time() - start_time))

programPause = input("Press the <ENTER> key to continue...")

