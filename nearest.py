import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import classification_report, confusion_matrix
import time

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

 # get two cifar 10 classes of the corresponding indexes
def get_two_classes(train_data, train_labels, test_data, test_labels, class_index_1, class_index_2):
    x_train = train_data[(train_labels == class_index_1) | (train_labels == class_index_2)]
    y_train = train_labels[(train_labels == class_index_1) | (train_labels == class_index_2)]
    x_test = test_data[(test_labels == class_index_1) | (test_labels == class_index_2)]
    y_test = test_labels[(test_labels == class_index_1) | (test_labels == class_index_2)]
    return x_train, y_train, x_test, y_test

    
if __name__ == "__main__":

    # path of the folder directory
    directory = 'cifar-10-batches-py'    

    x_train, y_train, x_test, y_test = load_data_from_directory(directory)

    # get airplane and bird classes in order to compare them with the SVM models
    x_train, y_train, x_test, y_test = get_two_classes(x_train, y_train, x_test, y_test, 0, 2)

    # print shape of train and test data
    print('Training data set shape for airplane - bird classes:', x_train.shape)
    print('Training label set shape for airplane - bird classes:', y_train.shape)
    print('Test data set shape for airplane - bird classes:', x_test.shape)
    print('Test label set shape for airplane - bird classes:', y_test.shape)

    start_time = time.time()

    # K-Nearest Neighbors classifier imported from sklearn
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)
    y_predknn = knn.predict(x_test)

    print("Nearest Neighbor for k=1\n")
    print(confusion_matrix(y_test, y_predknn), end="\n")
    print(classification_report(y_test, y_predknn), end="\n")

    print("--- Execution time for Nearest Neighbor (k=1) : %s seconds ---\n" % (time.time() - start_time))

    start_time = time.time()

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    y_predknn = knn.predict(x_test)

    print("Nearest Neighbor for k=3\n")
    print(confusion_matrix(y_test, y_predknn), end="\n")
    print(classification_report(y_test, y_predknn), end="\n")

    print("--- Execution time for Nearest Neighbor (k=3) : %s seconds ---\n" % (time.time() - start_time))

    start_time = time.time()

    # Nearest Centroid classifier imported from sklearn
    ncc = NearestCentroid()
    ncc.fit(x_train, y_train)
    y_predictncc=ncc.predict(x_test)

    print("Nearest Class Centroid\n")
    print(confusion_matrix(y_test, y_predictncc), end="\n")
    print(classification_report(y_test, y_predictncc), end="\n")

    print("--- Execution time for Nearest Class Centroid : %s seconds ---\n" % (time.time() - start_time))
