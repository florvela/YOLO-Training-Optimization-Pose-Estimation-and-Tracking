#!/bin/bash

################### install openvino ###################

# Step 1: Create a virtual environment for the notebooks
echo "Creating virtual environment for notebooks"
python3 -m venv .venv

# Step 2: Activate the virtual environment
echo "Activating virtual environment"
source .venv/bin/activate

# Step 3: Install OpenVINO and dependencies
echo "Installing OpenVINO and dependencies"
pip install --upgrade pip
pip install --no-deps openvino openvino-dev nncf
pip install -r requirements.txt


################### install gstreamer ###################

# Step 4: Install GStreamer and dependencies
echo "Installing GStreamer and dependencies"
sudo apt-get install gstreamer1.0*
sudo apt install ubuntu-restricted-extras
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Step 5: Install additional dependencies for GStreamer
echo "Installing additional dependencies for GStreamer"
pip install numpy

# Step 7: Clone and build OpenCV with GStreamer support
echo "Cloning and building OpenCV with GStreamer support"
git clone https://github.com/opencv/opencv.git

cd opencv/

git checkout 4.1.0

mkdir build

cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python3) \
-D BUILD_opencv_python2=OFF \
-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D WITH_GSTREAMER=ON \
-D BUILD_EXAMPLES=ON ..

sudo make -j$(nproc)

sudo make install

sudo ldconfig

# Step 8: Install GCC with a specific version
# echo "Installing GCC with a specific version"
# conda install -c conda-forge gcc=12.1.0