import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import random
import visualization


# Get the current working directory (project root)
project_root = os.path.abspath(os.path.dirname(__file__))  # Gets the directory of the script

# Define relative paths from the project root
noisy_dir = os.path.join(project_root, 'Noisy_Documents/noisyPadded')
clean_dir = os.path.join(project_root, 'Noisy_Documents/cleanPadded')
test_noisy_dir = os.path.join(project_root, 'Noisy_Documents/testCleanPadded')
test_clean_dir = os.path.join(project_root, 'Noisy_Documents/testNoisyPadded')


# Function to load and preprocess images
# Function to load and preprocess images
def load_images_from_directory(directory, target_size=(540, 420)):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = []
    
    for file in image_files:
        # Read image
        img = cv2.imread(os.path.join(directory, file), cv2.IMREAD_GRAYSCALE)  # Assuming grayscale
        # Resize image to target size
        img_resized = cv2.resize(img, target_size)
        images.append(img_resized)
    
    # Convert to numpy array and normalize
    images = np.array(images, dtype='float32') / 255.0  # Normalize between 0 and 1
    return images

# Load clean and noisy images
train_noisy = load_images_from_directory(noisy_dir)
train_clean = load_images_from_directory(clean_dir)
test_noisy = load_images_from_directory(test_noisy_dir)
test_clean = load_images_from_directory(test_clean_dir)

# Reshape the images to have 1 channel (for grayscale images)
train_noisy = np.expand_dims(train_noisy, axis=-1)  # Shape (num_samples, 540, 420, 1)
train_clean = np.expand_dims(train_clean, axis=-1)  # Shape (num_samples, 540, 420, 1)
test_noisy = np.expand_dims(test_noisy, axis=-1)    # Shape (num_samples, 540, 420, 1)
test_clean = np.expand_dims(test_clean, axis=-1)    # Shape (num_samples, 540, 420, 1)

# Define the convolutional autoencoder model
model = Sequential()

# Encoder
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(540, 420, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

# Latent representation (still spatial, just lower channel depth)
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

# Decoder
# model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))

# Output layer (reconstruct the image)
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))  # Output layer

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy')

# Train the model on noisy and clean images
model.fit(train_noisy, train_clean,
          epochs=2,
          batch_size=5,
          shuffle=True,
          validation_split=0.2)

# Predict the denoised images
decoded_imgs = model.predict(test_noisy)

# Display original clean image, noisy image, and denoised image
images = []


for i in range(3):
    images.append(test_clean[i])

for i in range(3):
    images.append(test_noisy[i])

for i in range(3):
    images.append(decoded_imgs[i])


# Use visualization function to display images
visualization.NOISYOFFICE_Output(images)
