# Semantic-Segmentation-with-ENet

![Tests](https://github.com/AshishAsokan/Semantic-Segmentation-with-ENet/workflows/E-Net/badge.svg)

Tensorflow 2.0 Implementation of the E-Net Semantic Segmentation Architecture

Based on the paper ["ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation"](https://arxiv.org/abs/1606.02147)

# Setup

- Clone the repository
- Install the package locally using the following command
```bash
pip install -e Semantic-Segmentation-with-Enet/
```

# Files in the Repository

- ```tests/tests_enet.py``` : Python script to test the dimensions of all the blocks and model using random data

- ```evaluation/dataset_prep.py``` :  Prepares the training data and decodes & encodes the segmented images in the dataset

- ```enet_seg/enet_blocks.py``` : Definitions of the Bottleneck layers in the architecture

- ```enet_seg/enet_model.py``` : Final model including all the layers outlined in the paper

- ```enet_seg/utilities.py``` : Contains the definition of the Max Unpool layer
