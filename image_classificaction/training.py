import os
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import cast

# Datos para entrenamiento
data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
data_train = cast(tf.data.Dataset, data['train'])
label_train = cast(list[str], metadata.features['label'].names)


def normalice(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


data_train = data_train.map(normalice)
data_train = data_train.cache()

# Creación de capas
entry_layer = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
hidden_layer1 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)
hidden_layer2 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)
output_layer = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
layers = [entry_layer, hidden_layer1, hidden_layer2, output_layer]

# Creación del modelo a partir de las capas
model = tf.keras.Sequential(layers=layers, name="clothing_classifier")
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
num_examples = metadata.splits["train"].num_examples
lot_size = 32
data_train = data_train.repeat().shuffle(num_examples).batch(lot_size)
training_result = model.fit(
    data_train, epochs=7, steps_per_epoch=math.ceil(num_examples/lot_size))

# Guardar el modelo
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ClothingClassifier.keras"
filepath = os.path.join(path, MODEL_NAME)
model.save(filepath=filepath)

# ------------ Gráfica del entrenamiento ------------
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.xlabel("# Época")
plt.ylabel("Magnitud de pérdida")
plt.plot(training_result.history["loss"])
plt.show()
