""" Entrenamiento del modelo de clasificación de ropa """
import os
import math
from typing import cast
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds  # pylint: disable=C0411

# Preparación de data
data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_data = cast(tf.data.Dataset, data['train'])
test_data = cast(tf.data.Dataset, data['test'])
categories = cast(list[str], metadata.features['label'].names)
train_data = train_data.map(map_func=lambda image, label: (
    tf.cast(image, tf.float32)/255, label))
test_data = test_data.map(map_func=lambda image, label: (
    tf.cast(image, tf.float32)/255, label))
num_examples = metadata.splits["train"].num_examples
LOTSIZE = 32
train_data = train_data.cache().repeat().shuffle(num_examples).batch(LOTSIZE)
test_data = test_data.cache().batch(LOTSIZE)

# Preparación de capas
entry_layer = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
hidden_layer1 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)
hidden_layer2 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)
output_layer = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
layers = [entry_layer, hidden_layer1, hidden_layer2, output_layer]

# Preparación del modelo
model = tf.keras.Sequential(layers=layers, name="clothing_classifier")
model.compile(optimizer=tf.optimizers.Adam(
), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
HISTORY = model.fit(train_data, epochs=10, steps_per_epoch=math.ceil(
    num_examples/LOTSIZE), validation_data=test_data)
model.evaluate(test_data)

# Guardar modelo
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ClothingClassifier.keras"
filepath = os.path.join(path, MODEL_NAME)
model.save(filepath=filepath)

# ------------ Gráfica del entrenamiento ------------
rows, columns = (1, 2)
WINDOW_TITLE = 'Training Result'
plt.figure(num=WINDOW_TITLE, figsize=(12, 4))

# Subgráfico 1:
accuracy, val_accuracy = (
    HISTORY.history['accuracy'], HISTORY.history['val_accuracy'])
plt.subplot(rows, columns, 1)
plt.title('Training and Validation Accuracy')
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Subgráfico 2:
loss, val_loss = (HISTORY.history['loss'], HISTORY.history['val_loss'])
plt.subplot(rows, columns, 2)
plt.title('Training and Validation Loss')
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplots_adjust(wspace=0.5)
plt.show()
