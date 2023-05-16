import os
from PIL import Image

# Set the path to the directory containing the images
image_dir = "../data/images/custom_images/images"

# Get a list of all the files in the directory
files = os.listdir(image_dir)

# Loop through each file in the directory
for file in files:
    # Check if the file is an image
    if file.endswith(".png") or file.endswith(".PNG") or file.endswith(".jpeg") or file.endswith(".JPEG") or file.endswith(".bmp") or file.endswith(".BMP") or file.endswith(".gif") or file.endswith(".GIF"):
        # Open the image using Pillow
        img = Image.open(os.path.join(image_dir, file))

        # Convert the image to RGB mode if it is in "P" mode
        img = img.convert("RGB")

        # Get the new file name by replacing the old extension with .jpg
        new_file_name = os.path.splitext(file)[0] + ".jpg"

        # Save the image as a .jpg file
        img.save(os.path.join(image_dir, new_file_name))

        # Delete the original file
        os.remove(os.path.join(image_dir, file))
