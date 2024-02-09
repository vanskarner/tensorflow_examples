import os
from typing import cast
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Preparación de data
data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
data_test = cast(tf.data.Dataset, data['test'])
categories_tests = cast(list[str], metadata.features['label'].names)


def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

lot_size = 25
data_test = data_test.map(normalize).batch(lot_size)

# Carga de modelo guardado
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ClothingClassifier.keras"
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)

# Ejecución de predicciones
firstBatch = next(iter(data_test))
images_test, labels_test = firstBatch
images_test = images_test.numpy()
labels_test = labels_test.numpy()
predicctions = model.predict(images_test)

# ------------ Gráfica de la predicción ------------


def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.get_cmap('binary'))

    etiqueta_prediccion = np.argmax(arr_predicciones)
    if etiqueta_prediccion == etiqueta_real:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(categories_tests[etiqueta_prediccion],
                                         100*np.max(arr_predicciones),
                                         categories_tests[etiqueta_real]),
               color=color)


def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(10), arr_predicciones, color="#777777")
    plt.ylim([0, 1])
    etiqueta_prediccion = np.argmax(arr_predicciones)

    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')


figsize = [5, 5]
predicctions = predicctions
labels_test = labels_test

filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
    plt.subplot(filas, 2*columnas, 2*i+1)
    graficar_imagen(i, predicctions, labels_test, images_test)
    plt.subplot(filas, 2*columnas, 2*i+2)
    graficar_valor_arreglo(i, predicctions, labels_test)
plt.show()
