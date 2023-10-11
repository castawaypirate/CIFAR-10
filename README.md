# CIFAR-10
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
<br />
<br />
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 
<br />
<br />
CIFAR-10 dataset was utilized to assess performance across Nearest Neighbor and Nearest Centroid methods, Convolutional Neural Networks (CNNs), Support Vector Machines (SVMs), and a custom Autoencoder, all implemented with TensorFlow and scikit-learn.
## Nearest Neighbor and Nearest Centroid
For the nearest neighbor classifier, the results were as follows:
<br />
`k=1:`
```
accuracy: 0.35
```
`k=3:`
```
accuracy: 0.33
```
For the nearest centroid classifier, the result was:
<br />
```
accuracy: 0.28
```
## Convolutional Neural Network
The architecture blends convolutional and dense layers from TensorFlow. We initiate convolutional layers at the beginning with 32, 64, 128, 256, 512, and 256 filters, each employing 3x3 kernels and 'same' padding to maintain input and output dimensions. To smoothly transition the data to the first fully connected layer, we reshape the output of the final convolutional layer using a flatten layer. The dense layers follow, comprising 256, 256, 128, and 10 neurons, respectively. The ultimate results are as follows:
<br />
`Train:`
```
accuracy: 0.94, loss: 0.17
```
`Test:`
```
accuracy: 0.88, loss: 0.35
```
## Support Vector Machines
SVM models from the Scikit-learn library (Support Vector Machines) were employed to classify a pair of classes from the Cifar-10 dataset, specifically "airplane" and "bird," using three different kernels: Linear, Polynomial, and Radial Basis Function (RBF). A grid search was conducted in order to determine the optimal C and gamma parameters. The final results were:
### Linear Kernel
**C = 0.01**
<br />
`Train:`
```
accuracy: 0.82
```
`Test:`
```
accuracy: 0.81
```
### Polynomial Kernel
**C = 1**
<br />
`Train:`
```
accuracy: 0.97
```
`Test:`
```
accuracy: 0.83
```
### Radial Basis Function Kernel
**C = 10, gamma = 0.01**
<br />
`Train:`
```
accuracy: 1.0
```
`Test:`
```
accuracy: 0.84
```
## Autoencoder
An Autoencoder wil be created with the aim of denoising the CIFAR-10 dataset. Subsequently, we will compare the results with the equivalent denoising process executed using PCA. The encoder architecture will comprise three convolutional layers with 64, 8, and 2 filters, each using 3x3 kernels. The decoder will consist of two transposed convolutional layers with 2 filters, also employing 3x3 kernels, and two additional convolutional layers with 32 and 3 filters, each using 3x3 kernels. In the case of PCA, we will retain 8 components from the original 3072 dimensions of the input data. The presented image illustrates the outcomes: the first row displays the original dataset images, the second row exhibits the same images with added noise, the third row showcases the denoised images reconstructed by the autoencoder, and the final row demonstrates the denoised images processed by PCA.
![noise 05 autoencoder 64 8 2 2 2 32 3 pca 8](https://github.com/castawaypirate/CIFAR-10/assets/32521649/bb0fb028-5b13-45ab-b026-736a3ec1ba79)

