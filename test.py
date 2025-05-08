import cv2
import os

# Define the directory containing the images
directory = '/Users/marcojonsson/AI_Class/Project-4/DenoisingAutoencoder/Noisy_Documents/clean/'

# List all image files in the directory
image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each image file and print its dimensions
for image_file in image_files:
    img_path = os.path.join(directory, image_file)
    
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    
    if img is not None:
        # Get the image dimensions
        height, width, channels = img.shape
        print(f"{image_file}: {width} x {height}")
    else:
        print(f"Error loading {image_file}")
