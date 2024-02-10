""" Entrenamiento del modelo de clasificación de ropa """
import os
import math
from typing import cast
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds # pylint: disable=C0411

# Preparación de data
data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
data_train = cast(tf.data.Dataset, data['train'])
label_train = cast(list[str], metadata.features['label'].names)
data_train = data_train.map(
    map_func=lambda image, label: (tf.cast(image, tf.float32)/255, label)
)
data_train = data_train.cache()

# Preparación de capas
entry_layer = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
hidden_layer1 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)
hidden_layer2 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)
output_layer = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
layers = [entry_layer, hidden_layer1, hidden_layer2, output_layer]

# Preparación del modelo
model = tf.keras.Sequential(layers=layers, name="clothing_classifier")
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
num_examples = metadata.splits["train"].num_examples
LOTSIZE = 32
data_train = data_train.repeat().shuffle(num_examples).batch(LOTSIZE)
TRAINING_RESULT = model.fit(
    data_train, epochs=7, steps_per_epoch=math.ceil(num_examples/LOTSIZE))

# Guardar modelo
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ClothingClassifier.keras"
filepath = os.path.join(path, MODEL_NAME)
model.save(filepath=filepath)

# ------------ Gráfica del entrenamiento ------------
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.xlabel("# Época")
plt.ylabel("Magnitud de pérdida")
plt.plot(TRAINING_RESULT.history["loss"])
plt.show()
