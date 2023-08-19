# YOLO Training and Optimization with Tracking and Pose Estimation

This repository contains the code for the training and optimization of YOLOv5, YOLOv7, and YOLOv8 models for object detection using Python 3. These models are widely used for real-time object detection tasks due to their accuracy and efficiency. The work involves training these models with a custom Roboflow dataset, and the use of optimization techniques utilizing OpenVINO and Apache TVM, adding an extra layer of performance. Additionally, the repository includes the implementation of pose estimation using yolov8-pose and tracking with ByteTrack.

The project is currently in development, and welcome to contributions and collaborations.

## 1) Datasets

This [directory](001-Datasets) contains code designed to facilitate the acquisition and preprocessing of gun detection datasets. 

The `create_datasets.ipynb` notebook assists in obtaining datasets from Roboflow. It provides functionality to download three distinct datasets:

1. **Dataset 1: Knives and Pistols**
   - Annotated images containing knives and pistols.
   
2. **Dataset 2: Guns**
   - Extensive dataset featuring the "gun" class.
   - Various images including people holding phones and guns of different sizes.
   
3. **Dataset 3: Randomized Clips (For Testing)**
   - Comprises images sourced from security cameras.
   - Primarily used for testing.


## 2) Training models

This [directory](002-Training-models) contains notebooks that focus on training YOLO (You Only Look Once) object detection models using different versions of the YOLO architecture.

### Notebooks

- [Train YOLOv5](002-Training-models/yolov5/train_yolov5.ipynb)
- [Train YOLOv7](002-Training-models/yolov7/train_yolov7.ipynb)
- [Train YOLOv8](002-Training-models/yolov8/train_yolov8.ipynb)

> **_NOTE:_**: To avoid memory issues in the display of the notebook use the following command: ```jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10```

## 3) Evaluating models

This [directory](003-Evaluating-models) contains notebooks designed for evaluating YOLO object detection models, specifically focusing on YOLOv5, YOLOv7, and YOLOv8 architectures. Additionally, the folder includes code for saving evaluation outputs to text files, converting them to CSV format, and facilitating a comprehensive understanding of the evaluation results.

### Notebooks

#### [Evaluate vs Original Dataset](eval_vs_original_dataset.ipynb)

In this notebook, you'll find instructions on evaluating YOLO models (YOLOv5, YOLOv7, and YOLOv8) on the original datasets they were trained on. The notebook provides guidance on loading the trained models and dataset, performing evaluation, and saving the evaluation output to a text file.

#### [Evaluate vs Randomized Clips](eval_vs_randomized_clips.ipynb)

Evaluate YOLO models (YOLOv5, YOLOv7, and YOLOv8) on the randomized clips dataset, which is intended for testing purposes. The notebook outlines the steps to load models, perform evaluation, and save the evaluation results to a text file.

#### [Understanding Output CSVs](understanding_output_csvs.ipynb)

This notebook delves into the process of reading and interpreting the CSV output files generated from the evaluation results. It explains the decisions and criteria used to select models for the final API, allowing for a clearer understanding of the evaluation outcomes.

## 4) Optimization with OpenVINO

This [directory](004-Optimization-with-OpenVINO) has the code for optimizing yolov8 models with OpenVINO. OpenVINO is a toolkit developed by Intel to optimize neural network models for efficient deployment across a wide range of hardware platforms, including CPUs, GPUs, and FPGAs. It provides pre-trained models, model conversion tools, and hardware-specific optimizations to accelerate inference performance while maintaining accuracy.

### Virtual Environment Setup

Create a virtual environment for the notebooks using the provided script:

```bash
#!/bin/bash
# Step 1: Create a virtual environment for the notebooks
echo "Creating virtual environment for notebooks" &&
python3 -m venv .openvino_venv &&

# Step 2: Activate the virtual environment
echo "Activating virtual environment" &&
source .openvino_venv/bin/activate &&

# Install necessary dependencies
pip install --upgrade pip &&
pip install --no-deps openvino openvino-dev nncf &&
pip install -r requirements.txt
```

### Optimizing YOLOv8 with OpenVINO

For optimizing the YOLOv8 model using OpenVINO, follow these steps:

1. Make sure you have the necessary YOLOv8 model checkpoint and configuration files prepared.

2. Adjust the file paths in the [main.py](004-Optimization-with-OpenVINO/main.py) script according to your case. 

3. Run the [main.py](004-Optimization-with-OpenVINO/main.py) script in your virtual environment, which you've set up using the provided instructions. This script should contain the logic to load the YOLOv8 model, perform OpenVINO optimization, and save the optimized models for deployment.


## 5) Optimization with Apache TVM

To test the optimization of the trained models using Apache TVM, follow these steps (from the [tutorial](https://tvm.apache.org/docs/install/from_source.html#)):

```bash
git clone --recursive https://github.com/apache/tvm tvm
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

cd tvm 
mkdir build
cp cmake/config.cmake build

conda env create --file conda/build-environment.yaml
conda activate tvm-build
```

modify build/config.cmake to add set(USE_LLVM /path/to/your/llvm/bin/llvm-config)

```bash
cd build
cmake .. -G Ninja
ninja

pip install -r requirements.txt

export TVM_HOME=/path/to/tvm
export TVM_HOME=~/Desktop/git_repos/YOLO-OpenVINO-TVM-GStreamer/005-Optimization-with-ApacheTVM/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

pip3 install --user numpy decorator attrs
pip3 install --user typing-extensions psutil scipy
pip3 install --user tornado
pip3 install --user tornado psutil 'xgboost>=1.1.0' cloudpickle

conda install -c conda-forge gcc=12.1.0
```

The code for optimizing yolo models can be found in the notebooks of this [directory](005-Optimization-with-ApacheTVM)

## Real-Time Tracking

The object tracking implementation in this project utilizes GStreamer.

### Setup

Build OpenCV 4 from source with Gstreamer:

```shell
./scripts/install_gstreamer.sh
```




## Contact

For any inquiries, suggestions, or collaborations related to this project, please feel free to reach out to flor.p.vela@gmail.com. I value your feedback and would be glad to assist you with any questions you may have.
