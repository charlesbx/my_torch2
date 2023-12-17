# My Torch Neural Network Script

## Overview

The `my_torch` script is a command-line tool for creating, training, and predicting with neural networks. It provides flexibility in creating new neural networks with specified architectures or loading existing ones. The script supports training the neural network using input data and expected outputs, as well as saving and loading the internal state of the neural network.

## Prerequisites

- Python (version 3.8 or higher)
- PyOpenCL

## Usage

```bash
./my_torch [--new IN_LAYER HIDDEN_LAYERS OUT_LAYER HIDDEN_LAYERS_NBR | --load LOADFILE] [--train | --predict] [--save SAVEFILE] FILE
```

## Options

- `--new`: Create a new neural network with random weights. Specify the architecture using the number of neurons in each layer.
  Example: `./my_torch --new 64 1024 4 1` creates a network with an input layer of 64 neurons, a hidden layer of 1024 neurons, and an output layer of 4 neurons.

- `--load`: Loads an existing neural network from `LOADFILE`.

- `--train`: Launches the neural network in training mode. Each board in `FILE` must contain inputs and expected outputs. Optionally, specify the number of epochs to train.

- `--predict`: Launches the neural network in prediction mode. Each board in `FILE` must contain inputs for the neural network.

- `--save`: Saves the neural network's internal state to `SAVEFILE`.

- `FILE`: A file containing inputs and possibly results for training or prediction.

## Examples

1. Create a new neural network:
    ```bash
    ./my_torch --new 64 1024 4 1 --save my_network.npz
    ```

2. Create a new neural network and train it:
    ```bash
    ./my_torch --new 64 1024 4 1 --train train_data.txt --save my_network.npz
    ```

3. Train the neural network:
    ```bash
    ./my_torch --load my_network.npz --train train_data.txt --save my_trained_network.npz
    ```

4. Make predictions using the trained network:
    ```bash
    ./my_torch --load my_trained_network.npz --predict test_data.txt
    ```