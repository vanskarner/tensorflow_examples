""" Exploración del set de datos usando solo tensorflow """
import matplotlib.pyplot as plt
import tensorflow as tf

train_data, test_data = tf.keras.datasets.fashion_mnist.load_data()
images, label_indexes = train_data
categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
              'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ------------ Mostrar primer elemento ------------
first_image, first_label = (images[0], categories[label_indexes[0]])
plt.suptitle('First Item')
plt.imshow(first_image, cmap='binary')
plt.xlabel(first_label)
plt.colorbar()
plt.grid(visible=False)
plt.show()

# ------------ Mostrar varios elementos ------------
rows, columns = (5, 5)
plt.figure(figsize=(10, 10))
plt.suptitle('Several Items')
for i in range(rows*columns):
    index = i+1
    plt.subplot(rows, columns, index)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap='binary')
    plt.xlabel(xlabel=categories[label_indexes[i]])
plt.show()
