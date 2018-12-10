import os
import time
import sys
import getopt

from itertools import chain
from fractions import Fraction

import numpy as np
from PIL import Image

try:
    from picamera import PiCamera
    import picamera.array
except ImportError:
    print("No Picamera")

WIDTH = 3280
HEIGHT = 2464
CUTOFF = 150


def log(data):
    with open('logs.txt', 'a') as fd:
        fd.write("%s\n" % data)


def get_image(camera):
    output = picamera.array.PiRGBArray(camera)
    time.sleep(5)
    # output = np.empty((WIDTH, HEIGHT, 3), dtype=np.uint8)
    camera.capture(output, "rgb")
    return output.array


def get_camera():
    camera = PiCamera(resolution=(WIDTH, HEIGHT), framerate=30)

    camera.resolution = (WIDTH, HEIGHT)
    camera.framerate = Fraction(1, 6)
    camera.shutter_speed = 3000000
    camera.exposure_mode = 'off'
    camera.iso = 800
    time.sleep(2)
    return camera


class Offset:
    # Los valores son sobre un eje X o Y
    value = 0  # Valor en cantidad de pixeles que se movio la imagen actual
    # Ademas se utiliza para ignorar los datos de los bordes en la alineacion
    # (Se podria separar en derecha/izquierda o arriba/abajo)
    diff = float("inf")  # Diferencia entre la referencia y la actual
    # Imagen de referencia
    reference = None


class ImgOffsets:
    # Define los valores para los dos ejes
    def __init__(self):
        self.x = Offset()
        self.y = Offset()

    def getTotalDiff(self):
        return self.x.diff + self.y.diff


class Stacker:
    def __init__(self):
        # Guardara el maximo de corrimiento que tuvo una
        self.cut_offset = ImgOffsets()

        # Total de imagenes procesadas (Para calcular el promedio)
        self.total_stacks = 1

        # Imagen resultado como una matriz de 3 dimensiones
        self.result = None

    def stack(self, im):
        # Apilo la imagen con el resultado previo
        offset = self.find_alignment(im)  # Busco la mejor alineacion

        # Muevo los valores (pixeles) para alinear la imagen
        im = np.roll(im, offset.x.value, axis=1)  # En el eje X (Ancho)
        im = np.roll(im, offset.y.value, axis=0)  # En el eje Y (Alto)

        # Si se movio la imagen una cantidad mayor a las veces previas
        # guardo la cantidad para ignorar el borde en los proximos apilamientos
        self.cut_offset.x.value = max(
            self.cut_offset.x.value, abs(offset.x.value))
        self.cut_offset.y.value = max(
            self.cut_offset.y.value, abs(offset.y.value))

        # Sumo la matriz resultado con la actual que fue alineada
        return np.add(self.result, im, dtype=np.uint32, out=self.result)

    def get_reference_array(self, img):
        # Obtengo los valores de los ejes
        img = np.clip(img, 128, None)  # Corto los colores mas oscuros
        # Me interesa los mas claros que es donde estan las estrellas
        # Si no lo hago el ruido en la oscuridad puede afectar
        # demaciado el resultado del alineamiento
        # Sumo los valores de los ejes y me queda una matriz de 2 dimensiones

        # Ancho x 3 colores
        arrayx = img.sum(axis=0, dtype=np.uint32)
        # Alto x 3 colores
        arrayy = img.sum(axis=1, dtype=np.uint32)

        return arrayx, arrayy

    def find_alignment(self, im):
        # Busco la mejor alineacion

        # Promedio de las imagenes de referencia (ya que se van sumando)
        referenceIm1x = self.cut_offset.x.reference // self.total_stacks
        referenceIm1y = self.cut_offset.y.reference // self.total_stacks

        # Tomo los valores de referencia para los ejes
        referenceIm2x, referenceIm2y = self.get_reference_array(im)

        bestOffset = ImgOffsets()  # Contiene el mejor resultado obtenido

        # Se intenta un alineamiento inicial sin desfasaje

        # Para eje X
        self.try_alignment(
            referenceIm1x, referenceIm2x, bestOffset.x, 0,
            self.cut_offset.x.value
        )

        # Para eje Y
        self.try_alignment(
            referenceIm1y, referenceIm2y, bestOffset.y, 0,
            self.cut_offset.y.value
        )

        # Se intenta alineamientos con un desfasaje de -9 a 9
        for off in chain(range(-CUTOFF, 0), range(1, CUTOFF)):
            # Para eje X
            self.try_alignment(
                referenceIm1x, referenceIm2x, bestOffset.x, off,
                self.cut_offset.x.value
            )
            # Para eje Y
            self.try_alignment(
                referenceIm1y, referenceIm2y, bestOffset.y, off,
                self.cut_offset.y.value
            )

        log("Aligned at: x:%s y: %s v: %s" % (
            bestOffset.x.value,
            bestOffset.y.value,
            bestOffset.getTotalDiff()))

        # Muevo los valores (pixeles) de las referencias de la imagen con el
        # mejor valor de alineamiento
        referenceIm2x = np.roll(referenceIm2x, bestOffset.x.value, axis=0)
        referenceIm2y = np.roll(referenceIm2y, bestOffset.y.value, axis=0)

        # Agrego los valores de referencia de la imagen al total global
        np.add(
            self.cut_offset.x.reference,
            referenceIm2x,
            dtype=np.uint32,
            out=self.cut_offset.x.reference,
        )
        np.add(
            self.cut_offset.y.reference,
            referenceIm2y,
            dtype=np.uint32,
            out=self.cut_offset.y.reference,
        )

        return bestOffset

    def try_alignment(self, im1, im2, bestOffsetAxis, offset, cutValue):
        # Pruevo el alineamiento con los valores dados
        diff = self.get_alignment(im1, im2, offset, cutValue)
        if diff < bestOffsetAxis.diff:
            # Si la diferencia es menor a todas las previas guardo su valor
            bestOffsetAxis.diff = diff
            bestOffsetAxis.value = offset

    def get_alignment(self, im1, im2, offset, cutValue):
        # Calculo cuanto de los bordes tengo que cortar
        toCutIm1S = max(cutValue, offset)
        toCutIm1E = max(cutValue, -offset)
        toCutIm2S = max(0, cutValue - offset)
        toCutIm2E = max(0, cutValue + offset)

        # Tomo las imagenes cortadas
        im1C = im1[toCutIm1S:im1.shape[0] - 1 - toCutIm1E, :]
        im2C = im2[toCutIm2S:im2.shape[0] - 1 - toCutIm2E, :]

        # Diferencia
        im_diff = np.subtract(im1C, im2C, dtype=np.int32)
        # Diferencia absoluta (positiva)
        im_diff = np.absolute(im_diff, out=im_diff)

        # Promedio de diferencia
        return im_diff.mean()

    def process_from_filesystem(self, path):
        i = 0
        # Busco todas las imagenes en la carpeta l que tengan extension .tif
        for file in os.listdir(path):
            if not file.endswith(".tif"):
                continue
            i += 1
            if i == 3:
                break

            filePath = os.path.join(path, file)
            log("Apilamiento %s con %s" % (self.total_stacks, filePath))

            start_time = time.time()  # Tomo el tiempo proceso para la imagen
            img = Image.open(filePath)  # Abro la imagen
            log("Read time: " + str(time.time() - start_time))

            start_time = time.time()  # Tomo el tiempo proceso para la imagen
            im_array = np.asarray(img)  # Tomo la matriz
            img.close()
            log("To Array time: " + str(time.time() - start_time))

            start_time = time.time()  # Tomo el tiempo proceso para la imagen
            # La primera imagen la guardo como resultado
            if self.result is None:
                cut_offset = self.cut_offset
                self.result = np.uint32(im_array)
                reference = self.get_reference_array(im_array)
                # Guardo los valores de referencia para los ejes
                cut_offset.x.reference, cut_offset.y.reference = reference
            else:
                # Inicio el apilamiento de la imagen
                self.result = self.stack(im_array)
                self.total_stacks += 1

            log("Process time: " + str(time.time() - start_time))

    def process_from_camera(self):
        camera = get_camera()

        # Busco todas las imagenes en la carpeta l que tengan extension .tif
        for i in range(5):
            log("Apilamiento: %s" % self.total_stacks)
            start_time = time.time()  # Tomo el tiempo proceso para la imagen

            im_array = get_image(camera)
            log("Got Image")

            # La primera imagen la guardo como resultado
            if self.result is None:
                cut_offset = self.cut_offset
                self.result = np.uint32(im_array)
                reference = self.get_reference_array(im_array)
                # Guardo los valores de referencia para los ejes
                cut_offset.x.reference, cut_offset.y.reference = reference
            else:
                # Inicio el apilamiento de la imagen
                self.result = self.stack(im_array)
                self.total_stacks += 1

            log("Elapsed time: " + str(time.time() - start_time))
            # if self.result is not None:
            #     self.save_image()

    def save_image(self):
        # Tomo el promedio (ya que se suman)
        result = np.uint8(self.result // self.total_stacks)
        # finalResult = np.clip(finalResult, 8, None)-8

        # Convierto la matriz en imagen
        image = Image.fromarray(result)

        # image = image.filter(
        #   ImageFilter.UnsharpMask(radius=2, percent=250, threshold=7))

        fname = time.strftime("%Y%m%d-%H%M%S") + ".tif"
        image.save(fname, "TIFF")
        image.close()
        log("Saved: %s" % fname)


def main():
    import cProfile
    import pstats
    from io import StringIO

    print('Argument List:', str(sys.argv))

    optlist = getopt.getopt(
        sys.argv, 'cpf:',  ['camera', 'profile', 'folder='])[0]
    # '--camera --profile --folder /lights'

    with_profile = False
    with_camera = False
    folder = '/'

    for o, a in optlist:
        if o in ("-p", "--profile"):
            with_profile = True
        elif o in ("-c", "--camera"):
            with_camera = True
        elif o in ("-f", "--folder"):
            folder = a

    pr = cProfile.Profile()
    if with_profile:
        pr.enable()

    stacker = Stacker()
    # Tomo el tiempo de ejecucion para el script
    start_time = time.time()
    if with_camera:
        stacker.process_from_camera()
    else:
        stacker.process_from_filesystem(folder)
    stacker.save_image()
    log("Total elapsed time: " + str(time.time() - start_time))

    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.dump_stats("states.ps")


if __name__ == "__main__":
    start = time.strftime("%Y%m%d-%H%M%S")
    log("Started at %s:" % start)
    main()
    end = time.strftime("%Y%m%d-%H%M%S")
    log("Ended at %s:\n" % end)
