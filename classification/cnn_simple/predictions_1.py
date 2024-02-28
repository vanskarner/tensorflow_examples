""" Predicciones del modelo de clasificación de ropa """
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Carga de modelo guardado
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "CIFAR10_Model1.keras"
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)

# Preparación de data
_, test_data = tf.keras.datasets.cifar10.load_data()
images, label_indexes = test_data
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
QUANTITY = 12
label_indexes = np.array(label_indexes).flatten()
label_indexes = label_indexes[:QUANTITY]
images = images[:QUANTITY]
images = images / 255.0

# Ejecución de predicciones
predicctions = model.predict(images)
predicted_percentages = [tf.nn.softmax(pred).numpy() for pred in predicctions]
max_predicted_percentages = [np.amax(pred) for pred in predicted_percentages]
predicted_label_indexes = [np.argmax(pred) for pred in predicctions]

# ------------ Gráfica de la predicción ------------
rows, columns = (3, 4)
plt.figure(figsize=(2*2*columns, 2*rows))
plt.suptitle('Predictions')
for i in range(QUANTITY):
    # Gráfico de la imagen y etiquetas
    real_label = categories[label_indexes[i]]
    predicted_label = categories[predicted_label_indexes[i]]
    predicted_percentage = f'{max_predicted_percentages[i]* 100:.2f}'
    plt.subplot(rows, 2*columns, 2*i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i])
    COLOR = 'blue' if real_label == predicted_label else 'red'
    xlabel = (f"{predicted_label} "
              f"{predicted_percentage}% "
              f"({real_label})")
    plt.xlabel(xlabel=xlabel, color=COLOR)

    # Gráfico de barras del arreglo de valores
    category_indexes = range(len(categories))
    category_predicted_percentages = predicted_percentages[i]
    predicted_label_index = predicted_label_indexes[i]
    real_label_index = label_indexes[i]
    plt.subplot(rows, 2*columns, 2*i+2)
    plt.grid(False)
    plt.xticks(category_indexes)
    plt.yticks([])
    bar_plot = plt.bar(
        category_indexes, category_predicted_percentages, color="#777777")
    plt.ylim([0, 1])
    bar_plot[predicted_label_index].set_color('red')
    bar_plot[real_label_index].set_color('blue')

plt.tight_layout()
plt.show()
