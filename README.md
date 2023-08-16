# YOLOv5, YOLOv7, and YOLOv8 Model Training and Optimization

This repository contains the code for training YOLOv5, YOLOv7, and YOLOv8 models for object detection using Python 3. These models are widely used for real-time object detection tasks due to their accuracy and efficiency. The models are trained on a custom Roboflow dataset. The project is currently in development, and welcome to contributions and collaborations.

## [001-Datasets](001-Datasets)

This directory contains code designed to facilitate the acquisition and preprocessing of gun detection datasets. 

The `create_datasets.ipynb` notebook assists in obtaining datasets from Roboflow. It provides functionality to download three distinct datasets:

1. **Dataset 1: Knives and Pistols**
   - Annotated images containing knives and pistols.
   
2. **Dataset 2: Guns**
   - Extensive dataset featuring the "gun" class.
   - Various images including people holding phones and guns of different sizes.
   
3. **Dataset 3: Randomized Clips (For Testing)**
   - Comprises images sourced from security cameras.
   - Primarily used for testing.


## [002-Training-models](002-Training-models)

This directory contains notebooks that focus on training YOLO (You Only Look Once) object detection models using different versions of the YOLO architecture.

### Notebooks

- [Train YOLOv5](002-Training-models/yolov5/train_yolov5.ipynb)
- [Train YOLOv7](002-Training-models/yolov7/train_yolov7.ipynb)
- [Train YOLOv8](002-Training-models/yolov8/train_yolov8.ipynb)

## Optimization

### OpenVINO

To optimize the trained YOLO models using OpenVINO, follow these steps:

1) Setup
> **_NOTE:_**: Requirements.txt is in the [scripts folder](https://github.com/florvela/YOLO-OpenVINO-TVM-GStreamer/tree/main/scripts)
```
# Step 1: Create a virtual environment for the notebooks
echo "Creating virtual environment for notebooks"
&& python3 -m venv .venv

# Step 2: Activate the virtual environment
&& echo "Activating virtual environment"
&& source .venv/bin/activate

# sudo apt install python3-pip
&& pip install --upgrade pip
&& pip install --no-deps openvino openvino-dev nncf
&& pip install -r requirements.txt
```
2) Try the [Hello World notebook](https://github.com/florvela/YOLO-OpenVINO-TVM-GStreamer/blob/main/002%20-%20Optimization%20with%20OpenVINO/notebooks/001-hello-world/001-hello-world.ipynb)
3) Optimize YOLO models
> **_NOTE:_**: To avoid memory issues in the display of the notebook use the following command: ```jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10```
5) Convert the trained models to OpenVINO Intermediate Representation (IR) format.

### Apache TVM

To test the optimization of the trained models using Apache TVM, follow these steps:

Install the Apache TVM framework on your machine. Refer to the official Apache TVM documentation for detailed instructions.
Use the TVM compiler to optimize the trained models.

## Real-Time Tracking

The object tracking implementation in this project utilizes GStreamer.

### Setup

Build OpenCV 4 from source with Gstreamer:

```shell
./scripts/install_gstreamer.sh
```




## Contact

For any inquiries, suggestions, or collaborations related to this project, please feel free to reach out to flor.p.vela@gmail.com. I value your feedback and would be glad to assist you with any questions you may have.
