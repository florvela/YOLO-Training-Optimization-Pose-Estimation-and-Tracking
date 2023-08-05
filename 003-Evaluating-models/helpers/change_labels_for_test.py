import os

def swap_first_zero_and_one(line):
    first_zero_index = line.find('0')
    first_one_index = line.find('1')

    if first_zero_index < first_one_index or first_one_index == -1:
        line = line[:first_zero_index] + '1' + line[first_zero_index + 1:]
    else:
        line = line[:first_one_index] + '0' + line[first_one_index + 1:]

    return line

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            processed_line = swap_first_zero_and_one(line)
            file.write(processed_line)

def process_directory(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            process_file(file_path)

labels_folder = "/content/drive/MyDrive/trabajo_final_CEIA/train_models/datasets/yolov8_big/test/labels"
process_directory(labels_folder)