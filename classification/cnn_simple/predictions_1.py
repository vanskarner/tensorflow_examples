""" Predicciones del modelo de clasificación de ropa """
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Preparación de data
_, test_data = tf.keras.datasets.cifar10.load_data()
images, labels = test_data
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
images = images / 255.0
QUANTITY = 12
labels = np.array(labels).flatten()
labels = labels[:QUANTITY]
images = images[:QUANTITY]

# Carga de modelo guardado
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "CIFAR10_Model1.keras"
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)

# Ejecución de predicciones
predicctions = model.predict(images)
predicted_percentages = [tf.nn.softmax(pred).numpy() for pred in predicctions]
predicted_percentages_max = [np.amax(pred) for pred in predicted_percentages]
predicted_labels = [categories[np.argmax(pred)] for pred in predicctions]

# ------------ Gráfica de la predicción ------------
# Gráfica 1
rows, columns = (3, 4)
plt.figure(figsize=(2*2*columns, 2*rows))
plt.suptitle('Predictions')
for i in range(QUANTITY):
    true_label = categories[labels[i]]
    plt.subplot(rows, columns, i + 1)
    plt.imshow(images[i])
    COLOR = 'green' if true_label == predicted_labels[i] else 'red'
    title = (f"True: {true_label}\n"
             f"Predicted: {predicted_labels[i]}\n"
             f"Percentage: {predicted_percentages_max[i]* 100:.2f}%")
    plt.title(title, color=COLOR)
    plt.axis('off')

plt.tight_layout()
plt.show()
