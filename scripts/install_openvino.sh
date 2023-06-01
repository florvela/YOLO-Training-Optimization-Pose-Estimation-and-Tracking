#!/bin/bash

#export PATH="/usr/local/bin:$PATH"

######## install openvino ########

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