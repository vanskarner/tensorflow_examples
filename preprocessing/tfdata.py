""" 
Carga y preprocesamiento de imágenes
Usando tf.data
"""
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ------------------------- OBTENER LA DATA -------------------------
# Descarga el archivo flower_photos.tgz desde la URL proporcionada
URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_dir = tf.keras.utils.get_file(
    origin=URL,
    fname='flower_photos',
    untar=True)

# Convierte el directorio de datos a un objeto pathlib.Path para facilitar su manipulación.
data_dir = pathlib.Path(data_dir)

# Cuenta el número total de imágenes en el directorio contando todos
# los archivos .jpg en todas las subcarpetas.
image_count = len(list(data_dir.glob('*/*.jpg')))

# Crea un array que contiene los nombres de todas las carpetas dentro del directorio
# excluyendo el elemento con el nombre "LICENSE.txt"
alguniterable = data_dir.glob('*')
names = [item.name for item in alguniterable if item.name != "LICENSE.txt"]
categories = np.array(sorted(names))

# Diccionario con la información relevante
info_data = {
    'data_dir': data_dir,
    'image_count': image_count,
    'categories': categories
}
# ------------------------ PREPARACIÓN DE DATASET ------------------------
# Crea un Dataset que contiene una lista de archivos.
list_ds = tf.data.Dataset.list_files(
    str(info_data['data_dir']/'*/*'), shuffle=False)

# Baraja aleatoriamente los elementos del conjunto de datos list_ds, el parámetro
# 'reshuffle_each_iteration' garantiza que la barajada no se realice en cada iteración.
list_ds = list_ds.shuffle(
    info_data['image_count'], reshuffle_each_iteration=False)

# Divide 'list_ds' en dataset de entrenamiento y validación, usando el 20%
# para validación. El conjunto de entrenamiento omite los primeros 'val_size'
# elementos, mientras que el de validación los toma.
val_size = int(info_data['image_count'] * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

batch_size = 32
img_height = 180
img_width = 180

AUTOTUNE = tf.data.AUTOTUNE


def get_label(file_path, class_names):
    """
    Obtiene la etiqueta para una imagen dada su ruta de archivo.

    Parámetros:
        file_path (str): La ruta de archivo de la imagen.

    Retorna:
        int: La etiqueta de la imagen, codificada como un entero.
    """
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)


def decode_img(img, size):
    """
    Decodifica una imagen comprimida en formato JPEG y la redimensiona al tamaño deseado.

    Parámetros:
        img (str): La imagen comprimida en formato JPEG.

    Retorna:
        tf.Tensor: La imagen decodificada y redimensionada como un tensor.
    """
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, size)


def process_path(file_path):
    """
    Procesa la ruta de archivo de una imagen, obteniendo su etiqueta y cargando la imagen.

    Parámetros:
        file_path (str): La ruta de archivo de la imagen.

    Retorna:
        tuple: Una tupla que contiene la imagen decodificada y redimensionada, y su etiqueta.
    """
    label = get_label(file_path, info_data['categories'])
    img = tf.io.read_file(file_path)
    img = decode_img(img, [img_height, img_width])
    return img, label


# Use Dataset.map para crear un conjunto de datos de pares de image, label :
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

image_batch, label_batch = next(iter(train_ds))

# ------------------------- MOSTRAR INFORMACION DEL DATASET CREADO -------------------------

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(info_data['categories'][label])
    plt.axis("off")
plt.show()

# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(len(class_names))
# ])

# model.compile(
#     optimizer='adam',
#     loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])

# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=3
# )
