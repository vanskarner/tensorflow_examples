""" Predicciones del modelo de conversión de Celsius a Fahrenheit"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Preparación de data
celsius_test = [-50, -28, 5, 35, 100]
fahrenheit_test = [-58, -18.4, 41, 95, 212]

# Carga de modelo guardado
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "CelToFah.keras"
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)

# Ejecución de predicciones
predictions = model.predict(x=celsius_test, use_multiprocessing=True)

# ------------ Gráfica de la predicción ------------
plt.figure(figsize=(8, 6))
plt.suptitle('Datos de prueba y predicciones del modelo')
plt.plot(celsius_test, fahrenheit_test, 'g-', label='Datos de prueba')
plt.plot(celsius_test, predictions, 'b-', label='Predicciones')
plt.xlim(21.44, 21.58)  # Limitamos el eje x
plt.ylim(70.64, 70.80)  # Limitamos el eje y
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.legend()
plt.grid(True)
plt.show()
