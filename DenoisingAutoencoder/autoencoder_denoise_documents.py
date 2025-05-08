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

# Paths to the noisy and clean images
noisy_dir = '/Users/marcojonsson/AI_Class/Project-4/DenoisingAutoencoder/Noisy_Documents/noisyPadded'
clean_dir = '/Users/marcojonsson/AI_Class/Project-4/DenoisingAutoencoder/Noisy_Documents/cleanPadded'

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

# Reshape the images to have 1 channel (for grayscale images)
train_noisy = np.reshape(train_noisy, (len(train_noisy), 540, 420, 1))
train_clean = np.reshape(train_clean, (len(train_clean), 540, 420, 1))

# Define the convolutional autoencoder model
model = Sequential()

# Encoder
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(540, 420, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

# Latent representation (still spatial, just lower channel depth)
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

# Decoder
model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))

# Output layer (reconstruct the image)
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))  # Output layer

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy')

# Train the model on noisy and clean images
model.fit(train_noisy, train_clean,
          epochs=3,
          batch_size=2,
          shuffle=True,
          validation_split=0.2)

# Predict the denoised images
decoded_imgs = model.predict(train_noisy)

# Display original clean image, noisy image, and denoised image
images = []

# Select a few images to display (you can change the range as needed)
for i in range(5):
    images.append(train_clean[i].reshape(540, 420))  # Original clean image
    images.append(train_noisy[i].reshape(540, 420))  # Noisy image
    images.append(decoded_imgs[i].reshape(540, 420))  # Denoised image

# Use visualization function to display images
visualization.displayListOfImgs(images)
