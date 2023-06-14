# YOLOv5, YOLOv7, and YOLOv8 Model Training and Optimization

This repository contains the code for training YOLOv5, YOLOv7, and YOLOv8 models for object detection using Python 3. These models are widely used for real-time object detection tasks due to their accuracy and efficiency. The models are trained on a custom Roboflow dataset. The project is currently in development, and welcome to contributions and collaborations.

## Setup

Build OpenCV 4 from source with Gstreamer:

```shell
./scripts/install_gstreamer.sh
```

## Training

1) Use the following notebooks in Google Colaboratory to train YOLO models using GPU:

    * [Training YOLO v5 on a custom dataset from Roboflow](https://github.com/florvela/YOLO-OpenVINO-TVM-GStreamer/blob/main/001%20-%20Training%20models/yolov5/train_yolov5.ipynb)
    * [Training YOLO v7 on a custom dataset from Roboflow](https://github.com/florvela/YOLO-OpenVINO-TVM-GStreamer/blob/main/001%20-%20Training%20models/yolov7/train_yolov7.ipynb)
    * [Training YOLO v8 on a custom dataset from Roboflow](https://github.com/florvela/YOLO-OpenVINO-TVM-GStreamer/blob/main/001%20-%20Training%20models/yolov8/train_yolov8.ipynb)

2) Make sure you save the wights of the trained models

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
2) Convert the trained models to OpenVINO Intermediate Representation (IR) format.

### Apache TVM

To test the optimization of the trained models using Apache TVM, follow these steps:

Install the Apache TVM framework on your machine. Refer to the official Apache TVM documentation for detailed instructions.
Use the TVM compiler to optimize the trained models.

## Real-Time Tracking

The object tracking implementation in this project utilizes GStreamer.

## Contact

For any inquiries, suggestions, or collaborations related to this project, please feel free to reach out to flor.p.vela@gmail.com. I value your feedback and would be glad to assist you with any questions you may have.
