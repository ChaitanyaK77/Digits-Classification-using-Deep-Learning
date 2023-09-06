# Digits-Classification-using-MLP


# MNIST Digit Classification using Sequential Neural Network with Keras

This repository showcases the implementation of a Sequential Neural Network (SNN) using Keras for the task of MNIST digit classification. The MNIST dataset is a well-known benchmark in the field of machine learning and computer vision, consisting of 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9. The goal of this project is to demonstrate the training process and the achieved accuracies of the SNN on this dataset.

## Installation

To run the code provided in this repository, you'll need to have Python and the following libraries installed:

```bash
pip install numpy tensorflow matplotlib
```

## Usage

The main code for training and evaluating the SNN can be found in the `mnist_digit_classification.ipynb` Jupyter Notebook. Simply run the notebook cell by cell to go through the training process, model compilation, and accuracy evaluation steps. The notebook provides detailed explanations for each step, making it suitable for both beginners and those familiar with neural networks.

## Model Architecture

The Sequential Neural Network (SNN) used in this project consists of multiple layers, including input, hidden, and output layers. The input layer has 784 nodes, corresponding to the flattened 28x28 pixel input images. The hidden layers employ ReLU activation functions for introducing non-linearity, and the output layer has 10 nodes, each representing one digit class, activated using the softmax function.

## Training and Accuracies

The SNN is trained using the training subset of the MNIST dataset. Various configurations, such as the number of hidden layers, the number of nodes in each hidden layer, the learning rate, and the batch size, have been experimented with to optimize the model's performance. The training process involves forward and backward propagation, weight updates using stochastic gradient descent, and the use of categorical cross-entropy as the loss function.

The model's performance is evaluated using the testing subset of the MNIST dataset. The accuracy achieved on the test data demonstrates the SNN's ability to generalize to unseen samples. The accuracy achieved may vary based on the specific architecture and hyperparameters used.

## Results

After experimenting with different configurations, the SNN achieved various accuracies on the test dataset. The results are as follows:
- Model 1: Accuracy of 92.88% with a no hidden layer.
- Model 2: Accuracy of 97.82% with one hidden layer


## Conclusion

This project demonstrates the process of building a Sequential Neural Network (SNN) using Keras for the classification of MNIST digits. By experimenting with different architectures and hyperparameters, we were able to achieve competitive accuracies on the test dataset. The code and insights provided here can serve as a foundation for further exploration and understanding of neural network models in image classification tasks.
