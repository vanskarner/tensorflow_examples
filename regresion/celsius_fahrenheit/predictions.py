""" Predicciones del modelo guardado de conversion de Celsius a Fahrenheit"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Datos para pruebas
celsius_test = [-50, -28, 5, 35, 100]
fahrenheit_test = [-58, -18.4, 41, 95, 212]

# Cargar modelo guardado
path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "CelToFah.keras"
filepath = os.path.join(path, MODEL_NAME)
model: tf.keras.Sequential = tf.keras.models.load_model(filepath=filepath)

# Ejecución de predicciones
predictions = model.predict(x=celsius_test, use_multiprocessing=True)

# ------------ Grafica de la predicción------------
plt.figure(figsize=(8, 6))
plt.title('Datos de prueba y predicciones del modelo')
plt.plot(celsius_test, fahrenheit_test, 'g-', label='Datos de prueba')
plt.plot(celsius_test, predictions, 'b-', label='Predicciones')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.legend()
plt.grid(True)
plt.show()
