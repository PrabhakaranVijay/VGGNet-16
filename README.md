# VGG-16 Image Classification Project

This project demonstrates the use of the VGG-16 Convolutional Neural Network (CNN) architecture for image classification. The model is trained on the ImageNet dataset and can classify images into 1000 different categories.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [References](#references)

---

## Introduction

VGG-16 is a popular deep learning model known for its simplicity and effectiveness. It uses 13 convolutional layers and 3 fully connected layers, making it suitable for large-scale image classification tasks.

This project uses a pre-trained VGG-16 model to predict the category of an input image.

---

## Requirements

To run this project, ensure you have the following dependencies installed. See [requirements.txt](requirements.txt) for details:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vgg16-image-classification.git
   cd vgg16-image-classification

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

## Usage

Download the project or clone it from GitHub.

Download a test image or use the default cat image included in the script:

    wget https://www.rd.com/wp-content/uploads/2021/01/GettyImages-1175550351.jpg -O cat_image.jpg

Run the script:
    
    python vgg16_classification.py
    
The script will:
- Load the VGG-16 model pre-trained on ImageNet.
- Preprocess the input image.
- Predict the top 3 categories and display their probabilities.
- Show the input image.

---

## Model Architecture

### Overview
The VGG-16 model consists of:
- Input: 224x224x3 RGB image
- 13 Convolutional Layers with 3x3 filters
- 5 Max Pooling Layers
- 3 Fully Connected Layers
- Softmax for classification into 1000 classes

---

## References

- VGG-16 Paper
- ImageNet Dataset
- TensorFlow Documentation: Link
- Keras Applications: Link

---
