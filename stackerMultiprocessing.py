import os
import time
from multiprocessing import Process
from multiprocessing import Pipe
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
logName = time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists('logs'):
    os.makedirs('logs')


def log(data):
    print(data)
    with open(f"logs/{logName}.txt", 'a') as fd:
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
    inputFolder = 'test'
    outputFolder = 'out/'

    def __init__(self):
        import sys
        import getopt
        optlist, _ = getopt.getopt(
            sys.argv[1:], 'cpi:o:',
            ['camera', 'profile', 'input=', 'output='])
        # '--camera --profile --input /lights'
        for o, a in optlist:
            if o in ("-p", "--profile"):
                self.with_profiling = True
            elif o in ("-c", "--camera"):
                self.with_camera = True
            elif o in ("-i", "--input"):
                self.inputFolder = a
            elif o in ("-o", "--output"):
                if a.endswith("/") or a == "":
                    self.outputFolder = a
                else:
                    self.outputFolder = a + "/"


class ImgReference:
    # Imagen de referencia
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __floordiv__(self, other):
        return ImgReference(self.x//other, self.y//other)

    def add(self, other):
        np.add(self.x, other.x, dtype=np.uint32, out=self.x)
        np.add(self.y, other.y, dtype=np.uint32, out=self.y)


class ImgCutoff:
    # Valor en cantidad de pixeles que se movio la imagen actual
    # Ademas se utiliza para ignorar los datos de los bordes en la alineacion
    # (Se podria separar en derecha/izquierda o arriba/abajo)
    def __init__(self):
        self.left = 0
        self.right = 0
        self.top = 0
        self.bottom = 0


class Stacker:
    def __init__(self):
        # Total de imagenes procesadas (Para calcular el promedio)
        self.total_stacks = 1

        # Imagen resultado como una matriz de 3 dimensiones
        self.result = None

        self.reference = None

        # Guardara el maximo de corrimiento que tuvo una
        self.cutoff = ImgCutoff()

        self.max_cutoff = 150

        self.max_dark_to_test = 128  # 128

    def stack(self, im):
        # Apilo la imagen con el resultado previo

        # Busco la mejor alineacion
        offset, im_reference = self.find_alignment(im)

        # Agrego los valores de referencia de la imagen al total global
        self.reference.add(im_reference)

        # Muevo los valores (pixeles) para alinear la imagen
        im = np.roll(im, offset.left, axis=1)  # En el eje X (Ancho)
        im = np.roll(im, offset.top, axis=0)  # En el eje Y (Alto)

        # Si se movio la imagen una cantidad mayor a las veces previas
        # guardo la cantidad para ignorar el borde en los proximos apilamientos
        self.cutoff.left = max(self.cutoff.left, abs(offset.left))
        self.cutoff.top = max(self.cutoff.top, abs(offset.top))

        # Sumo la matriz resultado con la actual que fue alineada
        return np.add(self.result, im, dtype=np.uint32, out=self.result)

    def get_reference_array(self, img):
        # Obtengo los valores de los ejes
        img = np.clip(img, self.max_dark_to_test, None)
        # Corto los colores mas oscuros
        # Me interesa los mas claros que es donde estan las estrellas
        # Si no lo hago el ruido en la oscuridad puede afectar
        # demaciado el resultado del alineamiento
        # Sumo los valores de los ejes y me queda una matriz de 2 dimensiones

        # Ancho x 3 colores
        arrayx = img.sum(axis=0, dtype=np.uint32)
        # Alto x 3 colores
        arrayy = img.sum(axis=1, dtype=np.uint32)

        return ImgReference(arrayx, arrayy)

    def find_alignment(self, im):
        # Busco la mejor alineacion

        # Promedio de las imagenes de referencia (ya que se van sumando)
        reference_im1 = self.reference // self.total_stacks

        # Tomo los valores de referencia para los ejes
        reference_im2 = self.get_reference_array(im)

        best_cutoff = ImgCutoff()  # Contiene el mejor resultado obtenido

        # Para eje X
        best_cutoff.left, discrepancy_x = self.try_alignment(
            reference_im1.x, reference_im2.x, self.cutoff.left
        )

        # Para eje Y
        best_cutoff.top, discrepancy_y = self.try_alignment(
            reference_im1.y, reference_im2.y, self.cutoff.top
        )

        log(f"""Aligned at: x: {best_cutoff.left} y: {best_cutoff.top}
        v: {discrepancy_x + discrepancy_y}""")

        # Muevo los valores (pixeles) de las referencias de la imagen con el
        # mejor valor de alineamiento
        reference_im2.x = np.roll(reference_im2.x, best_cutoff.left, axis=0)
        reference_im2.y = np.roll(reference_im2.y, best_cutoff.top, axis=0)

        return best_cutoff, reference_im2

    def try_alignment(self, im1, im2, cut_value):
        best_cutoff = 0  # Contiene el mejor resultado obtenido
        best_discrepancy = float("inf")
        max_cutoff = min(self.max_cutoff, len(im1)//8)

        # Se intenta alineamientos con un desfasaje de -max_cutoff a max_cutoff
        for off in range(-max_cutoff, max_cutoff):
            # Pruevo el alineamiento con los valores dados
            diff = self.get_alignment(im1, im2, off, cut_value)
            if diff < best_discrepancy:
                # Si la diferencia es menor a todas las previas guardo su valor
                best_discrepancy = diff
                best_cutoff = off
        return best_cutoff, best_discrepancy

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

        parent_conn, child_conn = Pipe()
        p = Process(target=self.get_images_from_path, args=(child_conn, path,))
        p.start()

        running = True
        while running:
            im_array = parent_conn.recv()
            if im_array is None:
                running = False
            else:
                # Tomo el tiempo proceso para la imagen
                start_time = time.time()
                self.process_im_array(im_array)
                log(f"Process time: {time.time() - start_time}")
        p.join()

    def get_images_from_path(self, conn, path):
        # Busco todas las imagenes en la carpeta l que tengan extension .tif
        for file in os.listdir(path):
            if not file.endswith(".tif") and not file.endswith(".TIF"):
                continue
            file_path = os.path.join(path, file)
            log(f"Apilamiento con {file_path}")

            start_time = time.time()  # Tomo el tiempo proceso para la imagen

            img = Image.open(file_path)  # Abro la imagen
            im_array = np.asarray(img)  # Tomo la matriz
            img.close()

            log(f"To Array: {time.time() - start_time}")

            conn.send(im_array)
        conn.send(None)
        conn.close()

    def process_from_camera(self):
        parent_conn, child_conn = Pipe()
        p = Process(target=self.get_images_from_camera, args=(child_conn,))
        p.start()

        running = True
        while running:
            im_array = parent_conn.recv()
            if im_array is None:
                running = False
            else:
                # Tomo el tiempo proceso para la imagen
                start_time = time.time()
                self.process_im_array(im_array)
                log(f"Process time: {time.time() - start_time}")
        p.join()

    def get_images_from_camera(self, conn):
        camera = get_camera()
        # Capturo imagenes desde la camara
        for _ in range(5):
            log("Capturando")
            start_time = time.time()  # Tomo el tiempo proceso para la imagen
            im_array = get_image(camera)
            log(f"Captura: {time.time() - start_time}")
            conn.send(im_array)
        conn.send(None)
        conn.close()

    def process_im_array(self, im_array):
        # La primera imagen la guardo como resultado
        if self.result is None:
            # Guardo los valores de referencia para los ejes
            self.reference = self.get_reference_array(im_array)
            self.result = np.uint32(im_array)
        else:
            # Inicio el apilamiento de la imagen
            self.result = self.stack(im_array)
            self.total_stacks += 1

    def save_image(self, output_folder):

        if self.result is None:
            raise Exception("No existe un resultado!")
        # Tomo el promedio (ya que se suman)
        result = np.uint8(self.result // self.total_stacks)
        # finalResult = np.clip(finalResult, 8, None)-8

        # Convierto la matriz en imagen
        image = Image.fromarray(result)

        # image = image.filter(
        #   ImageFilter.UnsharpMask(radius=2, percent=250, threshold=7))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        f_name = output_folder + time.strftime("%Y%m%d-%H%M%S") + ".tif"
        image.save(f_name, "TIFF")
        image.close()
        log(f"Saved: {f_name}")


def main():
    import cProfile
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
        stacker.process_from_filesystem(config.inputFolder)
    stacker.save_image(config.outputFolder)
    log(f"Total elapsed time: {time.time() - start_time}")

    if config.with_profiling:
        pr.disable()
        save_profile(pr)


def save_profile(profile):
    import pstats
    from io import StringIO
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
    ps.dump_stats("states.ps")


if __name__ == "__main__":
    start = time.strftime("%Y%m%d-%H%M%S")
    log(f"Started at: {start}")
    main()
    end = time.strftime("%Y%m%d-%H%M%S")
    log(f"Ended at: {end}\n")
