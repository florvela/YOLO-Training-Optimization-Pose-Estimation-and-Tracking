git clone --recursive https://github.com/apache/tvm tvm
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

cd tvm 
mkdir build
cp cmake/config.cmake build

conda env create --file conda/build-environment.yaml
conda activate tvm-build

# conda build --output-folder=conda/pkg  conda/recipe
# conda install tvm -c ./conda/pkg

cd build
cmake .. -G Ninja
ninja

pip install -r requirements.txt

export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

pip3 install --user numpy decorator attrs
pip3 install --user typing-extensions psutil scipy
pip3 install --user tornado
pip3 install --user tornado psutil 'xgboost>=1.1.0' cloudpickle