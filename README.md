# Handwritten Digit Recognition using Neural Network

This Python project implements a neural network using NumPy which is used to recognize handwritten digits.

## Training

The neural network is trained using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 training images and 10,000 test images, all of size 28x28 pixels. The training process involves forward propagation to compute predictions, backpropagation to compute gradients, and gradient descent to update the weights and biases of the network.

## Acknowledgements

This project is based on the code provided in the book "Neural Networks and Deep Learning" by Michael Nielsen (http://neuralnetworksanddeeplearning.com/).

## Installation and running the project

1. Clone this repository to your local machine and `cd` into it:

```bash
git clone https://github.com/ba-reynolds/handwritten-digits.git
cd handwritten-digits
```

2. Create a virtual environment, activate it and install the dependencies:
```bash
python3 -m venv venv
call venv/scripts/activate
pip install -r requirements.txt
```

3. Run `main.py`:
```bash
python3 src/main.py
```