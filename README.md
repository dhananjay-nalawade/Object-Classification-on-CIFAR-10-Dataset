
# Object-Classification-on-CIFAR-10-Dataset

Introduction :

The CIFAR (Canadian Institute for Advanced Research) dataset, comprises labeled images extensively employed in the realm of computer vision to assess and benchmark machine learning algorithms. The CIFAR dataset exists in two primary versions:

#1.  CIFAR-10: This dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

#2. CIFAR-100: This dataset contains 60,000 32x32 color images in 100 classes, with 600 images per class. The classes are grouped into 20 superclasses, each containing 5 subclasses.

The CIFAR dataset serves as a frequently employed benchmark for assessing image classification algorithms. Its popularity arises from its comparatively compact size and simplicity in contrast to larger datasets like ImageNet. This dataset is commonly utilized to gauge the effectiveness of deep learning models, especially convolutional neural networks (CNNs).

The CIFAR dataset has found application across diverse research domains, spanning object recognition, image segmentation, image restoration, and generative models. Various machine learning frameworks, such as TensorFlow and PyTorch, offer built-in functionalities for downloading and loading the CIFAR dataset, facilitating its use in training and testing machine learning models.


## Screenshots

![App Screenshot](https://miro.medium.com/v2/0*BdetXYemwXwOqNTs.jpg)


## Requirements
    pip install numpy
    pip install pandas
    pip install matplotlib
    pip install keras
## Load CIFAR10-DATASET

There are several ways to download and install the CIFAR dataset, depending on your needs and the machine learning framework you are using. Here are a few options:

#1. Using TensorFlow: If you are using TensorFlow, you can use the tf.keras.datasets.cifar10.load_data() function to download and load the CIFAR-10 dataset, or tf.keras.datasets.cifar100.load_data() function to download and load the CIFAR-100 dataset. These functions will automatically download the dataset and return it in a format that can be used for training and testing machine learning models.

#2. Using PyTorch: If you are using PyTorch, you can use the torchvision.datasets.CIFAR10 and torchvision.datasets.CIFAR100 classes to download and load the CIFAR-10 and CIFAR-100 datasets, respectively. These classes will automatically download the dataset and return it in a format that can be used for training and testing machine learning models.

#3. Downloading manually: If you prefer to download the dataset manually, you can visit the official CIFAR website at https://www.cs.toronto.edu/~kriz/cifar.html and download the dataset in either binary or text format. Once you have downloaded the dataset, you can use a script to convert the data into a format that can be used for machine learning.

Regardless of the method you choose, it's important to note that the CIFAR dataset is relatively large, and may take some time to download depending on your internet connection speed.
## Training (CNN tensorflow)


This is a Python code for training a convolutional neural network (CNN) on the CIFAR-10 dataset using TensorFlow Keras. Here is a summary of what the code does:

#1. Load the CIFAR-10 dataset using tf.keras.datasets.cifar10.load_data().

#2. Normalize the pixel values in the input images to be between 0 and 1.

#3. Print the first image in the training and test sets, along with their corresponding labels.

#4. Define a CNN architecture using tf.keras.models.Sequential() and add convolutional layers, pooling layers, and fully connected layers.

#5. Compile the model using the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.

#6. Train the model on the training set for 10 epochs and validate on the test set.
#7. Plot the training and validation accuracy over the epochs.

#8. Evaluate the final performance of the model on the test set by computing the test loss and accuracy using model.evaluate().

#9. Note that the code uses matplotlib for visualizing the images and the training/validation accuracy over epochs. It also assumes that TensorFlow Keras and matplotlib libraries are installed.