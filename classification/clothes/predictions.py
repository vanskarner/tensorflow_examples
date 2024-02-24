""" Predicciones del modelo de clasificación de ropa """
import os
from typing import cast
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds  # pylint: disable=C0411

# Cargar modelo guardado
MODEL_NAME = "ClothingClassifier.keras"
path = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)
model.summary()

# Preparación de data
data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
test_data = cast(tf.data.Dataset, data['test'])
categories_tests = metadata.features['label'].names
test_data = test_data.map(map_func=lambda image, label: (
    tf.cast(image, tf.float32)/255, label))
LOTSIZE = 25
num_examples = metadata.splits["test"].num_examples
test_data = test_data.cache().shuffle(num_examples).batch(LOTSIZE)

# Ejecución de predicciones
primerLote = test_data.take(1).as_numpy_iterator()
images_test, labels_test = next(primerLote)
predicctions = model.predict(images_test)

# for images_test, labels_test in test_data.take(1):
#     images_test = images_test.numpy()
#     labels_test = labels_test.numpy()
#     predicctions = model.predict(images_test)

# firstBatch = next(iter(test_data))
# images_test, labels_test = firstBatch
# images_test = images_test.numpy()
# labels_test = labels_test.numpy()
# predicctions = model.predict(images_test)

#------------ Gráfica de la predicción ------------
ROWS, COLUMNS = (5, 5)
NUM_IMAGES = ROWS*COLUMNS
plt.figure(figsize=(2*2*COLUMNS, 2*ROWS))
plt.suptitle('Predicciones')
for i in range(NUM_IMAGES):
    # Gráfico de la imagen y etiquetas
    plt.subplot(ROWS, 2*COLUMNS, 2*i+1)
    prediction, real_label, img = predicctions[i], labels_test[i], images_test[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[..., 0], cmap=plt.get_cmap('binary'))
    predicted_label = np.argmax(prediction)
    COLOR = 'blue' if predicted_label == real_label else 'red'
    xlabel = (f"{categories_tests[predicted_label]} "
              f"{100*np.max(prediction):2.0f}% "
              f"({categories_tests[real_label]})")
    plt.xlabel(xlabel=xlabel, color=COLOR)

    # Gráfico de barras del arreglo de valores
    plt.subplot(ROWS, 2*COLUMNS, 2*i+2)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    bar_plot = plt.bar(range(10), prediction, color="#777777")
    plt.ylim([0, 1])
    bar_plot[predicted_label].set_color('red')
    bar_plot[real_label].set_color('blue')
    # Agregar valores a las barras solo para predicted_label y real_label
    for j, val in enumerate(prediction):
        if j == predicted_label or j == real_label:
            plt.text(j, val, f'{val:.2f}', ha='center',
                     va='bottom', color='black')

plt.show()
