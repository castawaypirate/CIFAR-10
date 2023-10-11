import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

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

def reshape_data(data):
    data = data.reshape(len(data), 3, 32, 32)
    data = data.transpose(0, 2 ,3 ,1)
    return data


if __name__ == "__main__":

    # path of the folder directory
    directory = 'cifar-10-batches-py'    

    x_train, y_train, x_test, y_test = load_data_from_directory(directory)

    # print shape of train and test data
    print('Training data set shape:', x_train.shape)
    print('Training label set shape:', y_train.shape)
    print('Test data set shape:', x_test.shape)
    print('Test labl set shape:', y_test.shape)

    # reshape train and test data
    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)

    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        width_shift_range=0.15,
        height_shift_range=0.2
    )
    datagen.fit(x_train)

    # show image
    # image = x_train[0]
    # imgplot = plt.imshow(image)
    # plt.show()
    # plt.imshow(image)

    # print new shape of train and test data
    print('Training set shape:', x_train.shape)
    print('Test set shape:', x_test.shape)

    # normalize the train and test data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # one hot encoding to labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes = 10)

    # big test
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

    # small test
    # x_train, x_valid, y_train, y_valid = train_test_split(x_valid, y_valid, test_size=0.2)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_valid shape:', x_valid.shape)
    print('y_valid shape:', y_valid.shape)

    # model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation=LeakyReLU(0.3), padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation=LeakyReLU(0.3), padding='same'))
    model.add(Conv2D(128, (3, 3), activation=LeakyReLU(0.3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), activation=LeakyReLU(0.3), padding='same'))
    model.add(Conv2D(512, (3, 3), activation=LeakyReLU(0.3), padding='same'))
    model.add(Conv2D(256, (3, 3), activation=LeakyReLU(0.3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(256, activation=LeakyReLU(0.3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation=LeakyReLU(0.3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation=LeakyReLU(0.3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    # optimizer
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # compile model
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # fit data with and without image augmentation
    # results = model.fit(x_train,y_train, epochs=100, batch_size=128, verbose=2, validation_data=(x_valid, y_valid))
    results = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=125, batch_size=128, verbose=2, validation_data=(x_valid, y_valid))

    # model evaluation
    print("Train accuracy: ", model.evaluate(x_train, y_train, batch_size=128))
    print("Test accuracy: ", model.evaluate(x_test, y_test, batch_size=128))

    y_out = model.predict(x_test)

    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    y_pred = np.argmax(y_out, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred))
    model.summary()

    # show plots
    plt.figure(1)
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(['train loss', 'test loss'])
    plt.show()

    plt.figure(2)
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(['train accuracy', ' accuracy'])
    plt.show()
    
