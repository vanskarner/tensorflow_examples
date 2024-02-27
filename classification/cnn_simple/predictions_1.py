""" Predicciones del modelo de clasificación de ropa """
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Preparación de data
_, test_data = tf.keras.datasets.cifar10.load_data()
images, labels = test_data
categories_tests = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
images = images / 255.0

# Carga de modelo guardado
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "CIFAR10_Model2.keras"
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)

# Ejecución de predicciones
predicctions = model.predict(images)

labels = np.array(labels).flatten()
true_labels = [categories_tests[label] for label in labels[:25]]
predicted_labels = [categories_tests[np.argmax(
    pred)] for pred in predicctions[:25]]

# Calcular el porcentaje de cada categoría predicha
total_predictions = len(predicted_labels)
category_counts = {category: predicted_labels.count(
    category) for category in categories_tests}
category_percentages = {category: count / total_predictions *
                        100 for category, count in category_counts.items()}

# Mostrar las imágenes junto con las etiquetas verdaderas, predichas y sus porcentajes
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i])
    true_label = true_labels[i]
    predicted_label = predicted_labels[i]
    percentage = category_percentages[predicted_label]
    color = 'green' if true_label == predicted_label else 'red'
    plt.title(
        f'True: {true_label}\nPredicted: {predicted_label}\nPercentage: {percentage:.2f}%', color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
