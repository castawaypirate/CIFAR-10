import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# shuffle data
def shuffle(data, labels):
    idx = np.random.permutation(len(labels))
    data, labels = data[idx], labels[idx]
    return data, labels

# normalize data with the according method passed to the function
def normalization(train, test, method=None):
    train = train.astype('float32')
    test = test.astype('float32')
    if method == 'MinMax':
        scaler = MinMaxScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    elif method == 'Standard':
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    else:
        train = train / 255.0
        test = test / 255.0
    return train, test

# get two cifar 10 classes of the corresponding indexes
def get_two_classes(train_data, train_labels, test_data, test_labels, class_index_1, class_index_2):
    x_train = train_data[(train_labels == class_index_1) | (train_labels == class_index_2)]
    y_train = train_labels[(train_labels == class_index_1) | (train_labels == class_index_2)]
    x_test = test_data[(test_labels == class_index_1) | (test_labels == class_index_2)]
    y_test = test_labels[(test_labels == class_index_1) | (test_labels == class_index_2)]
    return x_train, y_train, x_test, y_test

# svm models for linear kernel
def svm_linear(x_train, y_train, x_test, y_test, c):
    print("-------------------------------------------------------------------------------------------------------")
    print("Linear Kernel")
    print("-------------------------------------------------------------------------------------------------------")
    linear = {}
    for C in c:
        clf = SVC(C=C, kernel='linear', max_iter=100000)
        clf.fit(x_train, y_train)
        linear[C] = [clf.score(x_train, y_train), clf.score(x_test, y_test), clf.support_vectors_.shape[0]]
        print('C = ' + str(C))
        print("Train accuracy: ", clf.score(x_train, y_train))
        print("Test accuracy: ", clf.score(x_test, y_test))
        print("Support Vectors: ", clf.support_vectors_.shape[0])
    return linear

# svm models for polynomial kernel
def svm_polynomial(x_train, y_train, x_test, y_test, c):
    print("-------------------------------------------------------------------------------------------------------")
    print("Polynomial Kernel")
    print("-------------------------------------------------------------------------------------------------------")
    poly = {}
    for C in c:
        clf = SVC(C=C, kernel='poly', degree=3, max_iter=100000)
        clf.fit(x_train, y_train)
        poly[C] = [clf.score(x_train, y_train), clf.score(x_test, y_test), clf.support_vectors_.shape[0]]
        print('C = ' + str(C))
        print("Train accuracy: ", clf.score(x_train, y_train))
        print("Test accuracy: ", clf.score(x_test, y_test))
        print("Support Vectors: ", clf.support_vectors_.shape[0])
    return poly

# svm models for rbf kernel
def svm_rbf(x_train, y_train, x_test, y_test, c, gamma):
    print("-------------------------------------------------------------------------------------------------------")
    print("RBF Kernel")
    print("-------------------------------------------------------------------------------------------------------")
    rbf = {}
    for G in gamma:
        for C in c:
            clf = SVC(C=C, kernel='rbf', gamma=G, max_iter=100000)
            clf.fit(x_train, y_train)
            parameters = tuple([G, C])
            rbf[parameters] = [clf.score(x_train, y_train), clf.score(x_test, y_test), clf.support_vectors_.shape[0]]
            print('gamma = ' + str(G) + ' C = ' + str(C))
            print("Train accuracy: ", clf.score(x_train, y_train))
            print("Test accuracy: ", clf.score(x_test, y_test))
            print("Support Vectors: ", clf.support_vectors_.shape[0])
    return rbf

# get the arrays of train and test accuracies along with the corresponding support vectors
def dictionary_to_arrays(dict):
    train = []
    test = []
    support_vectors = []
    for key in dict:
        train.append(dict[key][0])
        test.append(dict[key][1])
        support_vectors.append(dict[key][2])
    return train, test, support_vectors


if __name__ == "__main__":
    # path of the folder directory
    directory = 'cifar-10-batches-py'    
    x_train, y_train, x_test, y_test = load_data_from_directory(directory)
    train_x = x_train
    train_y = y_train
    test_x = x_test
    test_y = y_test

    # print shape of train and test data
    print('Training data set shape:', x_train.shape)
    print('Training label set shape:', y_train.shape)
    print('Test data set shape:', x_test.shape)
    print('Test label set shape:', y_test.shape)

    # airplane - bird test with split data set
    x_train, y_train, x_test, y_test = get_two_classes(x_train, y_train, x_test, y_test, 0, 2)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2)

    # print shape of train and test data
    print('Training data set shape for airplane - bird classes:', x_train.shape)
    print('Training label set shape for airplane - bird classes:', y_train.shape)
    print('Test data set shape for airplane - bird classes:', x_test.shape)
    print('Test label set shape for airplane - bird classes:', y_test.shape)

    # data shuffling
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    # normalization
    normalization_method = 'MinMax'
    print('Normalization method : ', normalization_method)
    x_train, x_test = normalization(x_train, x_test, normalization_method)

    # Linear Kernel
    c = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]
    linear = svm_linear(x_train, y_train, x_test, y_test, c)
    train, test, support_vectors = dictionary_to_arrays(linear)
    plt.figure(1)
    plt.title('Linear Kernel')
    plt.plot(support_vectors, train)
    plt.plot(support_vectors, test)
    plt.legend(['train accuracy', 'accuracy'])
    plt.show()

    # Polynomial Kernel
    c = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]
    poly = svm_polynomial(x_train, y_train, x_test, y_test, c)
    train, test, support_vectors = dictionary_to_arrays(poly)
    plt.figure(2)
    plt.title('Polynomial Kernel')
    plt.plot(support_vectors, train)
    plt.plot(support_vectors, test)
    plt.legend(['train accuracy', 'accuracy'])
    plt.show()

    # RBF Kernel
    c = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000] # 10 --> MinMax, 1 --> Standard, 1, 100 --> \255.0
    # c = [1]
    # gamma = [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]
    gamma = [0.01] # 0.01 --> MinMax, 0.001 --> Standard, 0.01 --> \255.0
    rbf = svm_rbf(x_train, y_train, x_test, y_test, c, gamma)
    train, test, support_vectors = dictionary_to_arrays(rbf)
    plt.figure(3)
    plt.title('RBF Kernel')
    plt.plot(support_vectors, train)
    plt.plot(support_vectors, test)
    plt.legend(['train accuracy', 'accuracy'])
    plt.show()


    # airplane - bird test with all the images of the two classes
    x_train, y_train, x_test, y_test = get_two_classes(train_x, train_y, test_x, test_y, 0, 2)

    # print shape of train and test data
    print('Training data set shape for airplane - bird classes:', x_train.shape)
    print('Training label set shape for airplane - bird classes:', y_train.shape)
    print('Test data set shape for airplane - bird classes:', x_test.shape)
    print('Test label set shape for airplane - bird classes:', y_test.shape)

    # data shuffling
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    # normalization
    normalization_method = 'MinMax'
    print('Normalization method : ', normalization_method)
    x_train, x_test = normalization(x_train, x_test, normalization_method)

    # Linear Kernel
    c = [1e-3, 0.01, 0.1]
    linear = svm_linear(x_train, y_train, x_test, y_test, c)
    train, test, support_vectors = dictionary_to_arrays(linear)
    plt.figure(1)
    plt.title('Linear Kernel')
    plt.plot(support_vectors, train)
    plt.plot(support_vectors, test)
    plt.legend(['train accuracy', 'accuracy'])
    plt.show()

    # Polynomial Kernel
    c = [0.1, 1, 10]
    poly = svm_polynomial(x_train, y_train, x_test, y_test, c)
    train, test, support_vectors = dictionary_to_arrays(poly)
    plt.figure(2)
    plt.title('Polynomial Kernel')
    plt.plot(support_vectors, train)
    plt.plot(support_vectors, test)
    plt.legend(['train accuracy', 'accuracy'])
    plt.show()

    # RBF Kernel
    # c = [1]
    # gamma = [1e-3, 0.01, 0.1]
    c = [0.1, 1, 10, 100]
    gamma = [0.01]
    rbf = svm_rbf(x_train, y_train, x_test, y_test, c, gamma)
    train, test, support_vectors = dictionary_to_arrays(rbf)
    plt.figure(3)
    plt.title('RBF Kernel')
    plt.plot(support_vectors, train)
    plt.plot(support_vectors, test)
    plt.legend(['train accuracy', 'accuracy'])
    plt.show()
