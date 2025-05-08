import cv2
import os

# Define the directory containing the images
directory = '/Users/marcojonsson/AI_Class/Project-4/DenoisingAutoencoder/Noisy_Documents/noisy/'
output_directory = '/Users/marcojonsson/AI_Class/Project-4/DenoisingAutoencoder/Noisy_Documents/noisyPadded/'

# List all image files in the directory
image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop through each image file to pad and save it
for image_file in image_files:
    img_path = os.path.join(directory, image_file)

    # Load the image using OpenCV
    img = cv2.imread(img_path)
    
    # Calculate the padding values
    top_bottom_padding = (420 - img.shape[0]) // 2
    bottom_padding = 420 - img.shape[0] - top_bottom_padding
    
    # Pad the image (zero padding)
    padded_img = cv2.copyMakeBorder(img, top_bottom_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Extract the name of the original file (without extension)
    file_name = os.path.splitext(image_file)[0]
    
    # Save the padded image with the original filename
    output_path = os.path.join(output_directory, f"{file_name}.png")  # You can adjust the extension if needed
    cv2.imwrite(output_path, padded_img)

    # Print the dimensions of the padded image
    print(f"Padded Image Dimensions: {padded_img.shape[1]} x {padded_img.shape[0]}")  # Should be 540x420
    print(f"Saved padded image as: {output_path}")
