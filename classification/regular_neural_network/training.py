"""
Entrenamiento del modelo
Usando tensorflow_datasets
"""

import os
import math
from typing import cast
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf

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
configmodel = {
    'name': 'clothing_classifier',
    'layers': layers,
    'optimizer': tf.optimizers.Adam(),
    'loss': tf.keras.losses.SparseCategoricalCrossentropy(),
    'metrics': ['accuracy'],
    'epochs': 10,
    'steps_per_epoch': math.ceil(num_examples/LOTSIZE),
    'train_data': train_data,
    'validation_data': test_data
}
model = tf.keras.Sequential(name=configmodel['name'],
                            layers=configmodel['layers'])
model.compile(optimizer=configmodel['optimizer'],
              loss=configmodel['loss'],
              metrics=configmodel['metrics'])
HISTORY = model.fit(
    x=configmodel['train_data'],
    epochs=configmodel['epochs'],
    steps_per_epoch=configmodel['steps_per_epoch'],
    validation_data=configmodel['validation_data'])
evaluation = model.evaluate(x=configmodel['validation_data'])
statisticsmodel = {
    'accuracy': HISTORY.history['accuracy'],
    'val_accuracy': HISTORY.history['val_accuracy'],
    'loss': HISTORY.history['loss'],
    'val_loss': HISTORY.history['val_loss'],
    'loss_evaluation': round(evaluation[0], 4),
    'accuracy_evaluation': round(evaluation[1], 4)
}

# Guardar modelo
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ClothingClassifier.keras"
filepath = os.path.join(path, MODEL_NAME)
model.save(filepath=filepath)

# ------------ Gráfica del entrenamiento ------------
rows, columns = (1, 2)
model_evaluation = f"""
Model evaluation:
loss: {statisticsmodel['loss_evaluation']} | accuracy: {statisticsmodel['accuracy_evaluation']}
"""
plt.figure(num='Training Result', figsize=(12, 6))
plt.suptitle(model_evaluation)

# Subgráfico 1:
plt.subplot(rows, columns, 1)
plt.title('Training and Validation Accuracy')
plt.plot(statisticsmodel['accuracy'], label='Training Accuracy')
plt.plot(statisticsmodel['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Subgráfico 2:
plt.subplot(rows, columns, 2)
plt.title('Training and Validation Loss')
plt.plot(statisticsmodel['loss'], label='Training Loss')
plt.plot(statisticsmodel['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
plt.show()
