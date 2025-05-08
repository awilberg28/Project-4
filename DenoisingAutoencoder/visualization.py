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
    samples = []
    for digit in range(10):
        # Get the first image that matches the current digit
        index = np.where(training_labels == digit)[0][0]
        samples.append(training_images[index])

    displayListOfImgs(samples)
    

def display_NoisyOffice_samples():
    clean_path = './Noisy_Documents/clean/'
    noisy_path = './Noisy_Documents/noisy/'

    image_pairs = []
    for i in range(3):
        clean_img_path = os.path.join(clean_path, f"{i}.png")
        noisy_img_path = os.path.join(noisy_path, f"{i}.png")

      
        clean_img = img_to_array(load_img(clean_img_path))
        noisy_img = img_to_array(load_img(noisy_img_path))

        image_pairs.append((clean_img / 255.0, noisy_img / 255.0)) 

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for col in range(3):
        axes[0, col].imshow(image_pairs[col][0])
        axes[0, col].set_title(f"Clean {col}.png")
        axes[0, col].axis('off')

        axes[1, col].imshow(image_pairs[col][1])
        axes[1, col].set_title(f"Noisy {col}.png")
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.show()