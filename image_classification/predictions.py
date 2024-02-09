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


def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas


data_test = data_test.map(normalizar)
data_test = data_test.cache()
TAMANO_LOTE = 25
data_test = data_test.batch(TAMANO_LOTE)

# Carga de modelo guardado
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ClothingClassifier.keras"
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)

# Ejecución de predicciones
firstBatch = next(iter(data_test))
imagenes_prueba, etiquetas_prueba = firstBatch
imagenes_prueba = imagenes_prueba.numpy()
etiquetas_prueba = etiquetas_prueba.numpy()
predicciones = model.predict(imagenes_prueba)

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
predicciones = predicciones
etiquetas_prueba = etiquetas_prueba

filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
    plt.subplot(filas, 2*columnas, 2*i+1)
    graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
    plt.subplot(filas, 2*columnas, 2*i+2)
    graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
plt.show()
