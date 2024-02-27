""" 
Carga y preprocesamiento de imágenes
Usando tf.data
"""
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ------------------------- OBTENCIÓN DE DATA -------------------------
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

# Crea un array que contiene los nombres de todas las carpetas dentro del directorio,
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

# Divide 'list_ds' en dataset de entrenamiento y validación, usando el 20% de los
# primeros para validación y el resto para entrenamiento.
val_size = int(info_data['image_count'] * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)


def get_label(file_path, class_names):
    """
    Obtiene la etiqueta para una imagen dada su ruta de archivo.

    Parámetros:
        file_path (str): La ruta de archivo de la imagen.
        class_names (list): Categorias en que se clasifican las imágenes.

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
        size (list): Dimensión de la imagen

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
    img_height, img_width = (180, 180)
    image_size = [img_height, img_width]
    image_label = get_label(file_path, info_data['categories'])
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_size)
    return img, image_label


# Aplica la función process_path a los datasets de entrenamiento y validación,
# asimismo las imagenes se cargan y procesan en paralelo
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# ------------------------ CONFIGURACIÓN DE DATASET ------------------------


def configure_for_performance(dataset):
    """
    Configura un conjunto de datos para un mejor rendimiento durante el entrenamiento del modelo.

    Parámetros:
        ds (tf.data.Dataset): El conjunto de datos a configurar.

    Retorna:
        tf.data.Dataset: El conjunto de datos configurado para un mejor rendimiento.
    """
    shuffle_buffer, batch_size = (1000, 32)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

# ------------------------ GRÁFICA DE DATASET ------------------------
image_batch, label_batch = next(iter(train_ds))
plt.figure(figsize=(10, 10))
plt.suptitle('Train DS')
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(info_data['categories'][label])
    plt.axis("off")
plt.show()

image_batch, label_batch = next(iter(val_ds))
plt.figure(figsize=(10, 10))
plt.suptitle('Validation DS')
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(info_data['categories'][label])
    plt.axis("off")
plt.show()

# ------------------------ ENTRENAMIENTO DE CNN CON EL DATASET ------------------------

# # Define un modelo de red neuronal convolucional utilizando la API Sequential de TensorFlow Keras
# model = tf.keras.Sequential([
#     # Normaliza los valores de píxeles de las imágenes al rango [0, 1] dividiendo por 255
#     tf.keras.layers.Rescaling(1./255),
#     # Capa convolucional con 32 filtros de tamaño 3x3 y función de activación ReLU
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     # Capa de agrupación máxima para reducir el tamaño de la imagen
#     tf.keras.layers.MaxPooling2D(),
#     # Repetición de las capas convolucionales y de agrupación para extraer características
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     # Capa que aplana la salida de la última capa convolucional para alimentarla a una capa densa
#     tf.keras.layers.Flatten(),
#     # Capa densa con 128 unidades y función de activación ReLU
#     tf.keras.layers.Dense(128, activation='relu'),
#     # Capa densa de salida con un número de unidades igual al número de clases en el conjunto de datos
#     tf.keras.layers.Dense(len(categories))
# ])

# # Compila el modelo con el optimizador 'adam', la función de pérdida y la métrica de precisión.
# model.compile(
#     optimizer='adam',
#     loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])

# # Entrena el modelo con datos de entrenamiento y validación durante 3 épocas.
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=3)

# # Evaluación del modelo
# model.evaluate(val_ds)
