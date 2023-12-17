#!/usr/bin/env python3
from src.NeuralNetwork import NeuralNetwork
from src.TrainingSet import TrainingSet
from src.args import Args
from sys import argv, stderr, exit
import numpy as np

def train_model(nn : NeuralNetwork, dataset, train_size=0.90, epochs=1, save_filename='model.npz'):
    ts = TrainingSet(dataset)
    X, y = ts.get_formatted_data()
    X_train, y_train = X[:int(len(X) * train_size)], y[:int(len(y) * train_size)]
    X_val, y_val = X[int(len(X) * train_size):], y[int(len(y) * train_size):]
    nn.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=epochs)
    nn.saveModel(save_filename)
    exit(0)

def predict(nn : NeuralNetwork, dataset):
    ts = TrainingSet(dataset)
    X, y = ts.get_formatted_data()
    import multiprocessing as mp
    predictions = []
    pool = mp.Pool()
    predictions = pool.map(nn.predict, X)
    pool.close()
    pool.join()
    print("Predictions:")
    predictions = [nn.format_output(prediction) for prediction in predictions]
    for i in range(len(X)):
        print("Prediction {}: {} - Expected: {}".format(i, predictions[i], y[i]))
    
    results = []
    for i in range(len(predictions)):
        if np.array_equal(predictions[i], y[i]):
            results.append(1)
        else:
            results.append(0)
    print("Accuracy: {}%".format(sum(results) / len(results) * 100))
    exit(0)

def create_model(save_filename, in_layer_neurons_nbr, out_layer_neurons_nbr, hidden_layers_nbr, hidden_layers_neurons_nbr):
    nn = NeuralNetwork(input_nbr=in_layer_neurons_nbr, output_nbr=out_layer_neurons_nbr, hidden_layers_nbr=hidden_layers_nbr, hidden_neurons_nbr=hidden_layers_neurons_nbr)
    nn.saveModel(save_filename)
    return nn

def main():
    argsManager = Args(argv)
    args = argsManager.args
    if args['help']:
        argsManager.help()
        exit(0)
    try:
        if args['new']:
            nn = create_model(args['save_filename'], args['in_layer_neurons_nbr'], args['out_layer_neurons_nbr'], args['hidden_layers_nbr'], args['hidden_layers_neurons_nbr'])
            if args['train']:
                train_model(nn, args['dataset_filename'], train_size=0.90, epochs=args['epochs'], save_filename=args['save_filename'])
        elif args['load']:
            if args['train']:
                train_model(NeuralNetwork(filename=args['load_filename']), args['dataset_filename'], train_size=0.90, epochs=args['epochs'], save_filename=args['load_filename'])
            elif args['predict']:
                predict(NeuralNetwork(filename=args['load_filename']), args['dataset_filename'])
        else:
            print("No action specified. Use -h or --help for more information.")
            exit(84)
    except Exception as e:
        print("An error occured: {}".format(e), file=stderr)
        exit(84)
    exit(0)

if __name__ == "__main__":
    main()