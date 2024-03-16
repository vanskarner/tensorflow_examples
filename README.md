# Ejemplos de TensorFlow
Conjunto de varios ejemplos simplificados sobre el uso tensorflow en cuestión de entrenamiento y predicción. Se actualizará regularmente para incluir nuevos ejemplos y mejoras.

# Motivación
Proporcionar una colección de ejemplos simplificados de TensorFlow que sirvan a modo de consulta para aquellos que deseen aprender o trabajar con esta potente biblioteca de aprendizaje automático. 

# Especificación
Ejecutado en Anaconda 23.9.0

- Python 3.10.13
- tensorflow 2.10.0
- tensorflow_datasets 4.9.4
- pip 23.3.1
- cuda-nvcc 12.3.107
- cudatoolkit 11.2.2
- cudnn 8.1.0
- numpy 1.26.4
- matplotlib 3.8.3

# Modelos Contenidos
## Regresión
### simple: Conversión de Celsius a Fahrenheit

<details>
<summary>Entrenamiento</summary>

![alt text](regresion/simple/Figure_Training.png)
</details>

<details>
<summary>Predicciones</summary>

![alt text](regresion/simple/Figure_Prediction.png)
</details>

## Clasificación
### cnn_simple: Clasificador de varias tipos usando el dataset de [`CIFAR-10`](https://www.tensorflow.org/datasets/catalog/cifar10?hl=es-419)

<details>
<summary>Entrenamiento</summary>

![alt text](classification/cnn_simple/Figure_Training.png)
</details>

<details>
<summary>Predicciones</summary>

![alt text](classification/cnn_simple/Figure_Prediction.png)
</details>

### regular_neural_network: Clasificador de ropa usando el dataset de [`Fashion MNIST`](https://github.com/zalandoresearch/fashion-mnist)

<details>
<summary>Entrenamiento</summary>

![alt text](classification/regular_neural_network/Figure_Training.png)
</details>

<details>
<summary>Predicciones</summary>

![alt text](classification/regular_neural_network/Figure_Prediction.png)
</details>
