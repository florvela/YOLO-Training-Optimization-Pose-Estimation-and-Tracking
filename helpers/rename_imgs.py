import os

directory = '../data/images/rifle'  # Change this to the directory you want to check

for filename in os.listdir(directory):
    if '?' in filename:
        new_filename = filename.split('?')[0]
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
