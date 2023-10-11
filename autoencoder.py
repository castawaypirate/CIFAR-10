import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

import tensorflow as tf

from keras import layers
from keras.models import Model
from keras.layers import LeakyReLU, Input

# code snippet from https://www.cs.toronto.edu/~kriz/cifar.html for python 3
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# loads data from directory
def load_data_from_directory(directory):

    train_data = None
    train_labels = []
    # for data batches 1 to 6
    for i in range(1, 6):
        batch = unpickle(directory + '/data_batch_' + str(i))
        if i == 1:
            train_data = batch[b'data']
        else:
            # merges two numpy arrays
            train_data = np.concatenate((train_data, batch[b'data']), axis=0)
        train_labels = train_labels + batch[b'labels']
    # from list to numpy array for train labels
    train_labels = np.array(train_labels)

    test_data = None
    test_labels = []

    batch = unpickle(directory + '/test_batch')
    test_data = batch[b'data']
    test_labels = batch[b'labels']

    # from list to numpy array for test labels
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels

# reshape data 
def reshape_data(data):
    data = data.reshape(len(data), 3, 32, 32)
    data = data.transpose(0, 2 ,3 ,1)
    return data

def rereshape_data(data):
    data = np.reshape(data,(len(data),32*32*3))
    return data

# show original, noisy and reconstructed images
def show(original, noisy, reconstructed, reconstructed_pca):
    fig = plt.figure(figsize=(10, 6))
    for i in range(5):
        fig.add_subplot(4, 5, i + 1)
        plt.imshow(original[i].reshape(32, 32, 3))

        fig.add_subplot(4, 5, i + 1 + 5)
        plt.imshow(noisy[i].reshape(32, 32, 3))

        fig.add_subplot(4, 5, i + 1 + 10)
        plt.imshow(reconstructed[i].reshape(32, 32, 3))

        fig.add_subplot(4, 5, i + 1 + 15)
        plt.imshow(reconstructed_pca[i].reshape(32, 32, 3))

    plt.show()

# autoencoder class
class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(32, 32, 3)),
      layers.Conv2D(64, (3, 3), activation=LeakyReLU(0.01), padding='same'),
      layers.BatchNormalization(),
      layers.Dropout(0.3),
      layers.Conv2D(8, (3, 3), activation=LeakyReLU(0.01), padding='same'),
      layers.BatchNormalization(),
      layers.Dropout(0.3),
      layers.Conv2D(2, (3, 3), activation=LeakyReLU(0.01), padding='same'),
      layers.BatchNormalization(),
      layers.Dropout(0.3)
    ])
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(2, (3, 3), activation=LeakyReLU(0.01), padding='same'),
      layers.Conv2DTranspose(2, (3, 3), activation=LeakyReLU(0.01), padding='same'),
      layers.Conv2D(32, (3, 3), activation=LeakyReLU(0.01), padding='same'),
      layers.BatchNormalization(),
      layers.Conv2D(3, (3, 3), activation=LeakyReLU(0.01), padding='same')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

# add noise to data for pca
def add_noise(data, noise_factor=0.5):
    noisy_data = data.copy()
    samples, dimensions = data.shape
    for i in range(samples):
        for j in range(dimensions):
            if np.random.rand() < noise_factor:
                noisy_data[i, j] = random.randint(0, 255)
    return noisy_data

if __name__ == "__main__":

    # path of the folder directory
    directory = 'cifar-10-batches-py'    

    # load data from local directory
    x_train, y_train, x_test, y_test = load_data_from_directory(directory)

    # big test
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

    # # small test
    # x_train, x_valid, y_train, y_valid = train_test_split(x_valid, y_valid, test_size=0.1)

    # print shape of train, valid and test data
    print('x_train shape:', x_train.shape)
    print('x_valid shape:', x_valid.shape)
    print('x_test shape:', x_test.shape)

    # add noise
    x_train_noisy = add_noise(x_train)
    x_valid_noisy = add_noise(x_valid)
    x_test_noisy = add_noise(x_test)

    # normalize data
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    x_train_noisy = x_train_noisy.astype('float32')
    x_valid_noisy = x_valid_noisy.astype('float32')
    x_test_noisy = x_test_noisy.astype('float32')
    x_train = x_train / 255.0
    x_valid = x_valid /255.0
    x_test = x_test / 255.0
    x_train_noisy = x_train_noisy / 255.0
    x_valid_noisy = x_valid_noisy /255.0
    x_test_noisy = x_test_noisy / 255.0

    # copy data for pca
    x_train_pca = x_train
    x_test_pca = x_test
    x_train_noisy_pca = x_train_noisy
    x_test_noisy_pca = x_test_noisy

    # print shape of pca data
    print('x_train_pca shape:', x_train_pca.shape)
    print('x_test_pca shape:', x_test_pca.shape)
    print('x_train_noisy_pca shape:', x_train_noisy_pca.shape)
    print('x_test_noisy_pca shape:', x_test_noisy_pca.shape)

    # reshape autoencoder data
    x_train = reshape_data(x_train)
    x_valid = reshape_data(x_valid)
    x_test = reshape_data(x_test)
    x_train_noisy = reshape_data(x_train_noisy)
    x_valid_noisy = reshape_data(x_valid_noisy)
    x_test_noisy = reshape_data(x_test_noisy)

    # print new shape of autoencoder data
    print('reshaped x_train shape:', x_train.shape)
    print('reshaped x_valid shape:', x_valid.shape)
    print('reshaped x_test shape:', x_test.shape)
    print('reshaped x_train_noisy shape:', x_train_noisy.shape)
    print('reshaped x_valid_noisy shape:', x_valid_noisy.shape)
    print('reshaped x_test_noisy shape:', x_test_noisy.shape)

    # # show images
    # image = x_train_noisy[0]
    # imgplot = plt.imshow(image)
    # plt.show()
    # plt.imshow(image)

    # autencoder
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(x_train_noisy, x_train,
                    batch_size=32,
                    epochs=50,
                    verbose=1,
                    validation_data=(x_valid_noisy, x_valid),
                    shuffle=True)

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    x_valid_reconstructed = autoencoder.predict(x_valid_noisy)
    x_test_reconstructed = autoencoder.predict(x_test_noisy)

    # pca
    pca = PCA(n_components=8)
    pca.fit(x_train_noisy_pca)
    x_test_reconstructed_pca = pca.inverse_transform(pca.transform(x_test_noisy_pca))

    # results
    x_test = rereshape_data(x_test)
    x_test_reconstructed = rereshape_data(x_test_reconstructed)
    mse = mean_squared_error(x_test, x_test_reconstructed)
    reconstruction_error = np.mean(np.abs(x_test - x_test_reconstructed))

    mse_pca = mean_squared_error(x_test_pca, x_test_reconstructed_pca)
    reconstruction_error_pca = np.mean(np.abs(x_test_pca - x_test_reconstructed_pca))

    print("Mean Squared Error (autoencoder):", mse)
    print("Reconstruction error (autoencoder):", reconstruction_error)

    print("Mean Squared Error (PCA):", mse_pca)
    print("Reconstruction error (PCA):", reconstruction_error_pca)

    x_test_pca = reshape_data(x_test_pca)
    x_test_noisy_pca = reshape_data(x_test_noisy_pca)
    x_test_reconstructed_pca = reshape_data(x_test_reconstructed_pca)

    show(x_test, x_test_noisy, x_test_reconstructed, x_test_reconstructed_pca)

    
