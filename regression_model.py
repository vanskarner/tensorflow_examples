""" modelo aa para predecir la conversion de Celsius a Fahrenheit """
import tensorflow as tf

# Data
celsius = [-40, -10, 0, 8, 15, 22]
fahrenheit = [-40, 14, 32, 46.4, 59, 71.6]
dataForTest = [100, 200, 300]

# Capas
entry_layer = tf.keras.layers.Flatten(input_shape=[1])
hidden_layer1 = tf.keras.layers.Dense(units=3)
hidden_layer2 = tf.keras.layers.Dense(units=3)
output_layer = tf.keras.layers.Dense(units=1)

# Modelo
layers = [entry_layer, hidden_layer1, hidden_layer2, output_layer]
model = tf.keras.Sequential(layers)
model.compile(
    optimizer=tf.optimizers.Adam(0.09),
    loss=tf.losses.MeanSquaredError()
)
training_result = model.fit(x=celsius, y=fahrenheit, epochs=150, verbose=False)

# Predicción
predictions = model.predict(x=dataForTest, use_multiprocessing=True)
print(predictions)

# ------------ Graficas ------------
import matplotlib.pyplot as plt

plt.title('Evolución de la pérdida durante el entrenamiento')
plt.xlabel('# Epocas')
plt.ylabel('# Magnitud de perdida')
plt.plot(training_result.history['loss'])
plt.show()
