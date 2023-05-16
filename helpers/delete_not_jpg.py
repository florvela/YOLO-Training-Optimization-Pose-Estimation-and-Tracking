import os

directory = '../data/images/rifle'  # Change this to the directory you want to check

for filename in os.listdir(directory):
    if not (filename.upper().endswith(".JPG") or filename.upper().endswith(".PNG") or filename.upper().endswith(".JPEG")):
        print(filename)
        os.remove(os.path.join(directory, filename))
