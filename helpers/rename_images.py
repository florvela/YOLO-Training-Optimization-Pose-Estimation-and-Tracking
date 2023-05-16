import os

# Set the path to the directory containing the images
image_dir = "../data/images/custom_images/images"

# Get a list of all the files in the directory
files = os.listdir(image_dir)

# Set a counter for the image numbers
count = 1

# Loop through each file in the directory
for file in files:
    # Check if the file is an image
    if file.endswith(".jpg"):
        # Get the new file name by adding the image number and .jpg extension
        new_file_name = "image_" + str(count) + ".jpg"

        # Rename the file
        os.rename(os.path.join(image_dir, file), os.path.join(image_dir, new_file_name))

        # Increment the counter
        count += 1
