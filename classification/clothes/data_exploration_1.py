""" Exploración del set de datos usando tensorflow_datasets """
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

data, metadata = tfds.load(
    name='fashion_mnist',
    as_supervised=True,
    with_info=True)
train_data = data['train']
categories = metadata.features['label'].names

# ------------ Mostrar primer elemento ------------
plt.suptitle('First Item')
for image, label_index in train_data.take(1):
    plt.imshow(image, cmap='binary')
    plt.xlabel(categories[label_index])
    plt.colorbar()
    plt.grid(visible=False)
    plt.show()
# ------------ Mostrar varios elemento ------------
rows, columns = (5, 5)
plt.figure(figsize=(10, 10))
plt.suptitle('Several Item')
for i, (image, label_index) in enumerate(train_data.take(25)):
    index = i+1
    plt.subplot(rows, columns, index)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap='binary')
    plt.xlabel(xlabel=categories[label_index])
plt.show()
