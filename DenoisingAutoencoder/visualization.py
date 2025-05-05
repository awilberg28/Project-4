import os
import numpy as np
import random
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.preprocessing.image import load_img, img_to_array


# Load MNIST dataset
(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

def display_MNIST_samples():
    """
    Displays one sample for each digit 0 through 9 from the MNIST training set.
    """
    samples = []
    for digit in range(10):
        # Get the first image that matches the current digit
        index = np.where(training_labels == digit)[0][0]
        samples.append(training_images[index])

    # Plotting the images
    fig, axes = plt.subplots(1, 10, figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i], cmap='gray')
        ax.set_title(f'{i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def display_NoisyOffice_samples():
    pass
    # ** YOUR CODE HERE **
    

display_MNIST_samples()
display_NoisyOffice_samples()