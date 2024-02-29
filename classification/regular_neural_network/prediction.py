"""
Predicciones del modelo 
Usando tensorflow_datasets
"""
import os
from typing import cast
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds  # pylint: disable=C0411

# Preparación de data
configdata = {
    'name': 'fashion_mnist',
    'batch_size': 25,
    'as_supervised': True,
    'with_info': True
}
data, metadata = tfds.load(
    name=configdata['name'],
    batch_size=configdata['batch_size'],
    as_supervised=configdata['as_supervised'],
    with_info=configdata['with_info'])
test_data = cast(tf.data.Dataset, data['test'])
num_examples = metadata.splits["test"].num_examples
test_data = test_data.cache().shuffle(num_examples)
firstBatch = test_data.take(1)
images, labels = next(firstBatch.as_numpy_iterator())
categories = metadata.features['label'].names

# Cargar modelo guardado y ejecutar predicciones
MODEL_NAME = "ClothingClassifier.keras"
path = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)
predicctions = model.predict(images)

# ------------ Gráfica de la predicción ------------
ROWS, COLUMNS = (5, 3)
NUM_IMAGES = ROWS*COLUMNS
plt.figure(figsize=(2*2*COLUMNS, 2*ROWS))
plt.suptitle('Predictions')
for i in range(NUM_IMAGES):
    # Gráfico de la imagen y etiquetas
    plt.subplot(ROWS, 2*COLUMNS, 2*i+1)
    prediction, real_label, img = predicctions[i], labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[..., 0], cmap='binary')
    predicted_label = np.argmax(prediction)
    COLOR = 'blue' if predicted_label == real_label else 'red'
    xlabel = (f"{categories[predicted_label]} "
              f"{100*np.max(prediction):2.0f}% "
              f"({categories[real_label]})")
    plt.xlabel(xlabel=xlabel, color=COLOR)

    # Gráfico de barras del arreglo de valores
    plt.subplot(ROWS, 2*COLUMNS, 2*i+2)
    plt.grid(False)
    plt.xticks(range(len(categories)))
    plt.yticks([])
    bar_plot = plt.bar(range(10), prediction, color="#777777")
    plt.ylim([0, 1])
    bar_plot[predicted_label].set_color('red')
    bar_plot[real_label].set_color('blue')
    # Agregar valores a las barras solo para predicted_label y real_label
    # for j, val in enumerate(prediction):
    #     if j == predicted_label or j == real_label:
    #         plt.text(j, val, f'{val:.2f}', ha='center',
    #                  va='bottom', color='black')

plt.tight_layout()
plt.show()
