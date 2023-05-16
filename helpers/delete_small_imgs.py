import os

root_directory = '../data/images'  # Change this to the root directory you want to check

for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        if os.path.getsize(filepath) < 1000:
            os.remove(filepath)
