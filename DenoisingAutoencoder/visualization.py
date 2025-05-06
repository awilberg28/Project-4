import os
import numpy as np
import random
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.preprocessing.image import load_img, img_to_array

(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

def displayListOfImgs(images):
    num_images = len(images)
    num_cols = round(num_images/2)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    if num_rows > 1 or num_cols > 1:axes = axes.flatten()
    else:axes = [axes]

    for (i,image) in enumerate(images):
        if i < len(axes):
            axes[i].imshow(image)
            axes[i].set_title(f"Image {i+1}")
            axes[i].axis('off')
    
    if num_images < len(axes):
        for i in range(num_images, len(axes)):fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def display_MNIST_samples():
    displayListOfImgs(training_set[0][:10])
    displayListOfImgs(testing_set[0][:10])

def display_NoisyOffice_samples():
    return

def display_NoisyOffice_samples():
    pass
    # ** YOUR CODE HERE **
    
>>>>>>> refs/remotes/origin/morgan

display_MNIST_samples()
display_NoisyOffice_samples()