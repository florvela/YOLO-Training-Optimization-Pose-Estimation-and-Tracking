sudo apt-get install gstreamer1.0*

&& sudo apt install ubuntu-restricted-extras

&& sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

&& conda create -n gstreamer_cv2 && conda activate gstreamer_cv2

&& pip install numpy

&& git clone https://github.com/opencv/opencv.git

&& cd opencv/

&& git checkout 4.1.0

&& mkdir build

&& cd build

&& cmake -D CMAKE_BUILD_TYPE=RELEASE \
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

&& sudo make -j$(nproc)

&& sudo make install

&& sudo ldconfig

&& conda install -c conda-forge gcc=12.1.0