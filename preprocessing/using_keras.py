import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

# ------------------------- OBTENCIÓN DE DATA -------------------------
# Descarga el archivo flower_photos.tgz desde la URL proporcionada
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)

# Convierte el directorio de datos a un objeto pathlib.Path para facilitar su manipulación.
data_dir = pathlib.Path(data_dir)

# ------------------------ PREPARACIÓN DE DATASET ------------------------
# Opciones para el tamaño del lote y el redimensionamiento de las imagenes
batch_size = 32
img_height = 180
img_width = 180

# Cargar conjunto de datos de imágenes desde un directorio para entrenamiento
# y aplica las opciones anteriores
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Cargar conjunto de datos de imágenes desde un directorio para validación
# y aplica las opciones anteriores
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Categorias en que se clasifican las imágenes.
categories = train_ds.class_names

# ------------------------ CONFIGURACIÓN DE DATASET ------------------------
# Para mejorar el rendimiento se utiliza las metodos de cache y prefetch
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# ------------------------ GRÁFICA DE DATASET ------------------------
# Muestra 9 elementos del dataset de entrenamiento
plt.figure(figsize=(10, 10))
plt.suptitle('Train DS')
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(categories[labels[i]])
        plt.axis("off")
plt.show()

# Muestra 9 elementos del dataset de validación
plt.figure(figsize=(10, 10))
plt.suptitle('Validation DS')
for images, labels in val_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(categories[labels[i]])
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
