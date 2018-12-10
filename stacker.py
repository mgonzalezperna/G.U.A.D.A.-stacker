import os
import time

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


class Config:

    with_profiling = False
    with_camera = False
    folder = '/'

    def __init__(self):
        import sys
        import getopt
        print('Argument List:', str(sys.argv))
        optlist, _ = getopt.getopt(
            sys.argv, 'cpf:',  ['camera', 'profile', 'folder='])
        # '--camera --profile --folder /lights'
        for o, a in optlist:
            if o in ("-p", "--profile"):
                self.with_profiling = True
            elif o in ("-c", "--camera"):
                self.with_camera = True
            elif o in ("-f", "--folder"):
                self.folder = a


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

        self.cutoff = 150

    def stack(self, im):
        # Apilo la imagen con el resultado previo

        # Busco la mejor alineacion
        offset, im_reference_x, im_reference_y = self.find_alignment(im)

        # Agrego los valores de referencia de la imagen al total global
        np.add(
            self.cut_offset.x.reference, im_reference_x,
            dtype=np.uint32, out=self.cut_offset.x.reference
        )
        np.add(
            self.cut_offset.y.reference, im_reference_y,
            dtype=np.uint32, out=self.cut_offset.y.reference
        )

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
        reference_im1_x = self.cut_offset.x.reference // self.total_stacks
        reference_im1_y = self.cut_offset.y.reference // self.total_stacks

        # Tomo los valores de referencia para los ejes
        reference_im2_x, reference_im2_y = self.get_reference_array(im)

        best_offset = ImgOffsets()  # Contiene el mejor resultado obtenido

        # Para eje X
        best_offset.x = self.try_alignment(
            reference_im1_x, reference_im2_x, self.cut_offset.x.value
        )

        # Para eje Y
        best_offset.y = self.try_alignment(
            reference_im1_y, reference_im2_y, self.cut_offset.y.value
        )

        log(f"""Aligned at:
        x: {best_offset.x.value} y: {best_offset.y.value}
        v: {best_offset.getTotalDiff()}""")

        # Muevo los valores (pixeles) de las referencias de la imagen con el
        # mejor valor de alineamiento
        reference_im2_x = np.roll(reference_im2_x, best_offset.x.value, axis=0)
        reference_im2_y = np.roll(reference_im2_y, best_offset.y.value, axis=0)

        return best_offset, reference_im2_x, reference_im2_y

    def try_alignment(self, im1, im2, cut_value):
        best_offset_axis = Offset()  # Contiene el mejor resultado obtenido

        # Se intenta alineamientos con un desfasaje de -cutoff a cutoff
        for off in range(-self.cutoff, self.cutoff):
            # Pruevo el alineamiento con los valores dados
            diff = self.get_alignment(im1, im2, off, cut_value)
            if diff < best_offset_axis.diff:
                # Si la diferencia es menor a todas las previas guardo su valor
                best_offset_axis.diff = diff
                best_offset_axis.value = off
        return best_offset_axis

    def get_alignment(self, im1, im2, offset, cut_value):
        # Calculo cuanto de los bordes tengo que cortar
        to_cut_im1_s = max(cut_value, offset)
        to_cut_im1_e = max(cut_value, -offset)
        to_cut_im2_s = max(0, cut_value - offset)
        to_cut_im2_e = max(0, cut_value + offset)

        # Tomo las imagenes cortadas
        im1_c = im1[to_cut_im1_s:im1.shape[0] - 1 - to_cut_im1_e, :]
        im2_c = im2[to_cut_im2_s:im2.shape[0] - 1 - to_cut_im2_e, :]

        # Diferencia
        im_diff = np.subtract(im1_c, im2_c, dtype=np.int32)
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

            file_path = os.path.join(path, file)
            log(f"Apilamiento {self.total_stacks} con {file_path}")

            start_time = time.time()  # Tomo el tiempo proceso para la imagen
            img = Image.open(file_path)  # Abro la imagen
            log(f"Read time: {time.time() - start_time}")

            start_time = time.time()  # Tomo el tiempo proceso para la imagen
            im_array = np.asarray(img)  # Tomo la matriz
            img.close()
            log(f"To Array time: {time.time() - start_time}")

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

            log(f"Process time: {time.time() - start_time}")

    def process_from_camera(self):
        camera = get_camera()

        # Busco todas las imagenes en la carpeta l que tengan extension .tif
        for _ in range(5):
            log(f"Apilamiento: {self.total_stacks}")
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

            log(f"Elapsed time: {time.time() - start_time}")
            # if self.result is not None:
            #     self.save_image()

    def save_image(self):

        if self.result is None:
            raise Exception("No existe un resultado!")
        # Tomo el promedio (ya que se suman)
        result = np.uint8(self.result // self.total_stacks)
        # finalResult = np.clip(finalResult, 8, None)-8

        # Convierto la matriz en imagen
        image = Image.fromarray(result)

        # image = image.filter(
        #   ImageFilter.UnsharpMask(radius=2, percent=250, threshold=7))

        f_name = time.strftime("%Y%m%d-%H%M%S") + ".tif"
        image.save(f_name, "TIFF")
        image.close()
        log(f"Saved: {f_name}")


def main():
    import cProfile
    import pstats
    from io import StringIO

    config = Config()

    pr = cProfile.Profile()
    if config.with_profiling:
        pr.enable()

    stacker = Stacker()
    # Tomo el tiempo de ejecucion para el script
    start_time = time.time()
    if config.with_camera:
        stacker.process_from_camera()
    else:
        stacker.process_from_filesystem(config.folder)
    stacker.save_image()
    log(f"Total elapsed time: {time.time() - start_time}")

    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.dump_stats("states.ps")


if __name__ == "__main__":
    start = time.strftime("%Y%m%d-%H%M%S")
    log(f"Started at: {start}")
    main()
    end = time.strftime("%Y%m%d-%H%M%S")
    log(f"Ended at: {end}\n")
