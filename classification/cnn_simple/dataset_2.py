""" Visualización de los datos """
from typing import cast
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf

data, metadata = tfds.load(
    name='cifar10',
    as_supervised=True,
    with_info=True,
    download=False)
test_data = cast(tf.data.Dataset, data['test'])
categories = metadata.features['label'].names

# ------------ Mostrar primer elemento ------------
plt.suptitle('First Image')
for first_image, first_label_index in test_data.take(1):
    plt.imshow(X=first_image)
    plt.xlabel(xlabel=categories[first_label_index])
    plt.colorbar()
    plt.grid(False)
    plt.show()

# ------------ Mostrar varios elemento ------------
plt.figure(figsize=(10, 10))
plt.suptitle('Several Images')
for i, (image, label_index) in enumerate(test_data.take(25)):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X=image)
    plt.xlabel(xlabel=categories[label_index])
plt.show()
