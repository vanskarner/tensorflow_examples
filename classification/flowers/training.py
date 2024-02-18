""" Entrenamiento del modelo """
import os
import matplotlib.pyplot as plt
import tensorflow as tf

(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Verifica los datos
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Crear la base convolucional
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
    32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))

model.summary()

# Compilar y entrenar el modelo.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

HISTORY = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Guardar modelo
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ClothingClassifier.keras"
filepath = os.path.join(path, MODEL_NAME)
model.save(filepath=filepath)

# Evaluar el modelo
plt.plot(HISTORY.history['accuracy'], label='accuracy')
plt.plot(HISTORY.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# 313/313 - 3s - loss: 0.8766 - accuracy: 0.7078 - 3s/epoch - 8ms/step
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
