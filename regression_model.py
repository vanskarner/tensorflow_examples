""" modelo aa para predecir la conversion de Celsius a Fahrenheit """
import matplotlib.pyplot as plt
import tensorflow as tf

# Data
dataset = {
    'train': {
        'x_axis': [-40, -10, 0, 8, 15, 22],
        'y_axis': [-40, 14, 32, 46.4, 59, 71.6]
    },
    'test': {
        'x_axis': [100, 200, 300],
        'y_axis': [212, 392, 572]
    }
}

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
TRAINING_RESULT = model.fit(x=dataset['train']['x_axis'], y=dataset['train']['y_axis'], epochs=150, verbose=False)

# Predicción
predictions = model.predict(x=dataset['test']['x_axis'], use_multiprocessing=True)
print(predictions)

# ------------ Graficas ------------

plt.title('Evolución de la pérdida durante el entrenamiento')
plt.xlabel('# Epocas')
plt.ylabel('# Magnitud de perdida')
plt.plot(TRAINING_RESULT.history['loss'])
plt.show()

plt.title('Precisión del modelo')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.scatter(dataset['test']['x_axis'], predictions, color='red',
            label='Predicciones del modelo')
plt.plot(dataset['test']['x_axis'], dataset['test']['y_axis'], 'bo-', label='Valores reales')
plt.legend()
plt.show()
