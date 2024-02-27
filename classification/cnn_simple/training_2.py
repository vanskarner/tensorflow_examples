"""
Entrenamiento del modelo
Segunda forma usando tensorflow_datasets
https://www.tensorflow.org/datasets/catalog/overview?hl=es-419
"""
import os
from typing import cast
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf

# Preparación de data
data, metadata = tfds.load(
    'cifar10',
    as_supervised=True,
    with_info=True)
train_data = cast(tf.data.Dataset, data['train'])
test_data = cast(tf.data.Dataset, data['test'])
categories = metadata.features['label'].names


def normalize_image(image, label):
    """ Normalizacion de datos """
    return tf.cast(image, tf.float32) / 255.0, label


train_data = train_data.map(normalize_image)
test_data = test_data.map(normalize_image)
BATCH_SIZE = 64
train_data = train_data.cache().shuffle(1000).batch(
    BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_data = test_data.cache().batch(BATCH_SIZE)

# Preparación de capas
layer1 = tf.keras.layers.Conv2D(
    32, (3, 3), activation='relu', input_shape=(32, 32, 3))
layer2 = tf.keras.layers.MaxPooling2D((2, 2))
layer3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
layer4 = tf.keras.layers.MaxPooling2D((2, 2))
layer5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
layer6 = tf.keras.layers.Flatten()
layer7 = tf.keras.layers.Dense(64, activation='relu')
layer8 = tf.keras.layers.Dense(10)
layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8]

# Preparación del modelo
model = tf.keras.models.Sequential(layers=layers, name='CIFAR10_Model')
model.summary()
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# Entrenamiento del modelo
HISTORY = model.fit(train_data, epochs=10,
                    validation_data=test_data)


# Evaluación del modelo
evaluation = model.evaluate(test_data, verbose=2)
loss_evaluation = round(evaluation[0], 4)
accuracy_evaluation = round(evaluation[1], 4)

# Guardar el modelo
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "CIFAR10_Model2.keras"
filepath = os.path.join(path, MODEL_NAME)
model.save(filepath=filepath)

# ------------ Gráfica del entrenamiento ------------
rows, columns = (1, 2)
model_evaluation = f"""
Model evaluation:
loss: {loss_evaluation} | accuracy: {accuracy_evaluation}
"""
plt.figure(num='Training Result', figsize=(12, 6))
plt.suptitle(model_evaluation)

# Subgráfico 1
plt.subplot(rows, columns, 1)
plt.title('Training and Validation Accuracy')
plt.plot(HISTORY.history['accuracy'], label='Training Accuracy')
plt.plot(HISTORY.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Subgráfico 2
plt.subplot(rows, columns, 2)
plt.title('Training and Validation Loss')
plt.plot(HISTORY.history['loss'], label='Training Loss')
plt.plot(HISTORY.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
plt.show()
