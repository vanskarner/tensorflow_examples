""" Predicciones del modelo de clasificación de ropa """
import os
from typing import cast
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Preparación de data
data, metadata = tfds.load('cifar10', as_supervised=True, with_info=True)
data_test = cast(tf.data.Dataset, data['test'])
categories_tests = cast(list[str], metadata.features['label'].names)
LOTSIZE = 25
data_test = data_test.map(
    map_func=lambda image, label: (tf.cast(image, tf.float32)/255, label)
).batch(LOTSIZE)

# Carga de modelo guardado
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "CIFAR10_Model2.keras"
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)

# Ejecución de predicciones
firstBatch = next(iter(data_test))
images_test, labels_test = firstBatch
images_test = images_test.numpy()
labels_test = labels_test.numpy()
predicctions = model.predict(images_test)

true_labels = [categories_tests[label] for label in labels_test[:25]]
predicted_labels = [categories_tests[np.argmax(pred)] for pred in predicctions[:25]]

# Calcular el porcentaje de cada categoría predicha
total_predictions = len(predicted_labels)
category_counts = {category: predicted_labels.count(category) for category in categories_tests}
category_percentages = {category: count / total_predictions * 100 for category, count in category_counts.items()}

# Mostrar las imágenes junto con las etiquetas verdaderas, predichas y sus porcentajes
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images_test[i])
    true_label = true_labels[i]
    predicted_label = predicted_labels[i]
    percentage = category_percentages[predicted_label]
    color = 'green' if true_label == predicted_label else 'red'
    plt.title(f'True: {true_label}\nPredicted: {predicted_label}\nPercentage: {percentage:.2f}%', color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
