"""
Visualización de los datos
Primera forma usando solo los datasets de tensorflow
"""
import matplotlib.pyplot as plt
import tensorflow as tf

data_train, data_test = tf.keras.datasets.cifar10.load_data()
images, label_indexes = data_train
categories = ['airplane', 'automobile', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ------------ Mostrar primer elemento ------------
first_image = images[0]
first_label = categories[label_indexes[0][0]]

plt.suptitle('First Image')
plt.imshow(X=first_image)
plt.xlabel(xlabel=first_label)
plt.colorbar()
plt.grid(False)
plt.show()

# ------------ Mostrar varios elemento ------------
plt.figure(figsize=(10, 10))
plt.suptitle('Several Images')
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(categories[label_indexes[i][0]])
plt.show()
