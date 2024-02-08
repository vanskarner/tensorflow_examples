""" Predicciones del modelo guardado """
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Datos de prueba
celsius_test = [-50, -28, 5, 35, 100]
fahrenheit_test = [-58, -18.4, 41, 95, 212]

# Cargar modelo
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "CelToFah.keras"
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)

# Predicciones
predictions = model.predict(x=celsius_test, use_multiprocessing=True)

# ------------ Graficas ------------
plt.figure(figsize=(8, 6))
plt.scatter(celsius_test, fahrenheit_test, c='r', label='Datos de prueba')
plt.plot(celsius_test, predictions, 'b-', label='Predicciones')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.title('Datos de prueba y predicciones del modelo')

# Agregar etiquetas de texto en los puntos de las predicciones
for c, f in zip(celsius_test, predictions):
    plt.text(c, f, f'{f[0]:.1f}', ha='right', va='bottom', fontsize=8)

plt.legend()
plt.grid(True)
plt.show()