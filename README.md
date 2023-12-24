# Deep Learning Projects

This repository contains two deep learning projects utilizing Kaggle datasets for image classification tasks.

## Project 1: MNIST Digit Recognition

### Dataset
The project utilizes the MNIST dataset sourced from Kaggle, containing grayscale images of hand-written digits (0-9).

### Overview
This project aims to predict the numbers depicted in the images using a neural network architecture. The implemented model comprises 3 layers: an input layer, 1 hidden layer, and an output layer.

### Implementation
- **Dataset**: MNIST dataset
- **Model Architecture**: 3-layer neural network
  - Input Layer
  - Hidden Layer
  - Output Layer
- **Training Process**: Utilized techniques like backpropagation and stochastic gradient descent.
- **Accuracy**: Achieved an impressive 99% accuracy, demonstrating effective digit recognition.

### Usage
1. Clone the repository.
2. Download the MNIST dataset from Kaggle and place it in the specified directory.
3. Install the necessary dependencies.
4. Run the provided notebook/script to train and evaluate the model.

## Project 2: ASL Alphabet Identification

### Dataset
The project utilizes the ASL (American Sign Language) alphabet dataset from Kaggle, comprising images representing individual letters in the ASL alphabet.

### Overview
This project focuses on identifying ASL alphabet letters using deep learning techniques. Data augmentation methods were employed to enrich the dataset and improve model accuracy. Strategies to prevent overfitting were implemented, including the utilization of multiple layers.

### Implementation
- **Dataset**: ASL alphabet dataset from Kaggle
- **Data Augmentation**: Techniques applied to increase dataset size
- **Model Architecture**: Employed multiple layers (e.g., Convolutional layers, Dropout) to minimize overfitting.
- **Training Process**: Utilized techniques like batch normalization and early stopping.
- **Accuracy**: Achieved over 98% accuracy on both training and validation data.

### Usage
1. Clone the repository.
2. Download the ASL alphabet dataset from Kaggle and place it in the specified directory.
3. Install the necessary dependencies.
4. Run the provided notebook/script to preprocess, train, and evaluate the ASL alphabet identification model.

## Requirements
Ensure the required libraries and their versions (e.g., TensorFlow, Keras, PyTorch, etc.) are listed in the requirements.txt file.

## Conclusion
Both projects showcased successful image classification tasks using deep learning models. The MNIST project attained 99% accuracy in predicting digits, while the ASL alphabet identification project achieved over 98% accuracy on both training and validation datasets.

## Contributors
Project done while taking the NVIDIA Intro to Deep Learning Course

