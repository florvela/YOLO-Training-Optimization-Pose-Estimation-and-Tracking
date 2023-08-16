# Gun Detection Code

Welcome to the Gun Detection Code folder. This directory contains code designed to facilitate the acquisition, preprocessing, and testing of gun detection datasets. The code is organized into the following sections:

## Datasets

The `create_datasets.ipynb` notebook assists in obtaining datasets from Roboflow. It provides functionality to download three distinct datasets:

1. **Dataset 1: Knives and Pistols**
   - Annotated images containing knives and pistols.
   
2. **Dataset 2: Guns**
   - Extensive dataset featuring the "gun" class.
   - Various images including people holding phones and guns of different sizes.
   
3. **Dataset 3: Randomized Clips (For Testing)**
   - Comprises images sourced from security cameras.
   - Primarily used for testing gun detection models.

## Testing Enhancement

The `test_enhancement.ipynb` notebook focuses on improving testing relevance. It downloads the randomized clips dataset and modifies bounding box labels to align with the first version of the dataset. This alignment facilitates effective testing of models trained on version 1.

## Usage

To utilize the provided code, simply open the respective Jupyter notebooks using your preferred environment and follow the instructions within. Make sure to have the required dependencies installed.

## Support

If you encounter any issues or have questions, feel free to reach out for assistance.

