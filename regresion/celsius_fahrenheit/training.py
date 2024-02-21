""" Entrenamiento del modelo de conversión de Celsius a Fahrenheit """
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Preparación de data
celsius_train = [-40, -10, 0, 8, 15, 22]
fahrenheit_train = [-40, 14, 32, 46.4, 59, 71.6]

# Preparación de capas
entry_layer = tf.keras.layers.Flatten(input_shape=[1])
hidden_layer1 = tf.keras.layers.Dense(units=3)
hidden_layer2 = tf.keras.layers.Dense(units=3)
output_layer = tf.keras.layers.Dense(units=1)
layers = [entry_layer, hidden_layer1, hidden_layer2, output_layer]

# Preparación del modelo
model = tf.keras.Sequential(layers, name='celsius_fahrenheit')
model.compile(
    optimizer=tf.optimizers.Adam(0.09),
    loss=tf.losses.MeanSquaredError()
)
history: tf.keras.callbacks.History = model.fit(
    x=celsius_train,
    y=fahrenheit_train,
    epochs=150,
    verbose=False)

# Guardar modelo
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "CelToFah.keras"
filepath = os.path.join(path, MODEL_NAME)
model.save(filepath=filepath)

# ------------ Gráfica del entrenamiento ------------
plt.xlabel('# Epocas')
plt.ylabel('# Magnitud de pérdida')
plt.plot(history.history['loss'])
plt.show()
