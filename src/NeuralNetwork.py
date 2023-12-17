import numpy as np
import random
import math
from os import remove, mkdir
import pyopencl as cl
import pyopencl.array as cl_array
import multiprocessing as mp
from shutil import rmtree
import time

class NeuralNetwork:
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 0.01
    
    def __init__(self, input_nbr=64, output_nbr=4, hidden_neurons_nbr=1024, hidden_layers_nbr=1, filename=None, loss_function='mse'):
        self.last_accuracy = None
        self.last_loss = None
        self.last_val_accuracy = None
        self.last_val_loss = None
        if loss_function == 'binary_crossentropy':
            self.loss_function = self.binary_crossentropy
        elif loss_function == 'mse':
            self.loss_function = self.mse
        else:
            raise Exception('Loss function not found')
        self.accuracy = []
        self.loss = []
        self.val_accuracy = []
        self.val_loss = []
        if filename is not None:
            self.loadModel(filename)
            return
        self.input_nbr = input_nbr
        self.output_nbr = output_nbr
        self.hidden_neurons_nbr = hidden_neurons_nbr
        self.hidden_layers_nbr = hidden_layers_nbr
        self.hidden_layers = []
        self.output_layer = []
        self.output_layer_error = []
        self.hidden_layers_error = []
        self.output_layer_weights = []
        self.hidden_layers_weights = []
        self.hidden_layers_bias = []
        self.output_layer_bias = []
        self.dropout = []
        
        self.init_weights()
        self.init_bias()
        self.init_dropout()
    
    def forward_propagation_gpu(self, input):
        self.hidden_layers = []
        self.output_layer = []
        for i in range(self.hidden_layers_nbr):
            self.hidden_layers.append([])
            if i == 0:
                self.hidden_layers[i] = np.zeros(self.hidden_neurons_nbr)
                for j in range(self.hidden_neurons_nbr):
                    a_dev = cl_array.to_device(self.queue, input)
                    b_dev = cl_array.to_device(self.queue, np.array(self.hidden_layers_weights[i][j], dtype=np.float32))
                    c_dev = cl_array.empty_like(a_dev)
                    self.program.multiply(self.queue, a_dev.shape, None, a_dev.data, b_dev.data, c_dev.data)
                    sum = c_dev.get()
                    
                    a_dev = cl_array.to_device(self.queue, sum)
                    sum_dev = cl_array.to_device(self.queue, np.array([0], dtype=np.float32))
                    self.program.sum(self.queue, a_dev.shape, None, a_dev.data, sum_dev.data)
                    self.hidden_layers[i][j] = sum_dev.get()[0]
            else:
                self.hidden_layers[i] = np.zeros(self.hidden_neurons_nbr)
                for j in range(self.hidden_neurons_nbr):
                    a_dev = cl_array.to_device(self.queue, self.hidden_layers[i-1])
                    b_dev = cl_array.to_device(self.queue, np.array(self.hidden_layers_weights[i][j], dtype=np.float32))
                    c_dev = cl_array.empty_like(a_dev)
                    self.program.multiply(self.queue, a_dev.shape, None, a_dev.data, b_dev.data, c_dev.data)
                    self.hidden_layers[i][j] = np.sum(c_dev.get())
            
            a_dev = cl_array.to_device(self.queue, self.hidden_layers[i])
            b_dev = cl_array.to_device(self.queue, np.array(self.hidden_layers_bias[i], dtype=np.float32))
            c_dev = cl_array.empty_like(a_dev)
            self.program.add(self.queue, a_dev.shape, None, a_dev.data, b_dev.data, c_dev.data)
            self.hidden_layers[i] = c_dev.get()
            
            a_dev = cl_array.to_device(self.queue, self.hidden_layers[i])
            self.program.sigmoid(self.queue, a_dev.shape, None, a_dev.data)
            self.hidden_layers[i] = a_dev.get()
            for j in range(self.hidden_neurons_nbr):
                if random.uniform(0, 1) < self.DROPOUT_RATE:
                    self.hidden_layers[i][j] = 0

        self.output_layer = np.zeros(self.output_nbr)
        for i in range(self.output_nbr):
            a_dev = cl_array.to_device(self.queue, self.hidden_layers[-1])
            b_dev = cl_array.to_device(self.queue, np.array(self.output_layer_weights[i], dtype=np.float32))
            c_dev = cl_array.empty_like(a_dev)
            self.program.multiply(self.queue, a_dev.shape, None, a_dev.data, b_dev.data, c_dev.data)
            sum = c_dev.get()
            
            a_dev = cl_array.to_device(self.queue, sum)
            sum_dev = cl_array.to_device(self.queue, np.array([0], dtype=np.float32))
            self.program.sum(self.queue, a_dev.shape, None, a_dev.data, sum_dev.data)
            self.output_layer[i] = sum_dev.get()[0]
            self.output_layer[i] += self.output_layer_bias[i]

        a_dev = cl_array.to_device(self.queue, self.output_layer)
        self.program.sigmoid(self.queue, a_dev.shape, None, a_dev.data)
        self.output_layer = a_dev.get()

    def back_propagation_gpu(self, input, expected_output):
        self.output_layer_error = []
        self.hidden_layers_error = []
        
        # compute output layer error
        a_dev = cl_array.to_device(self.queue, expected_output)
        b_dev = cl_array.to_device(self.queue, self.output_layer)
        c_dev = cl_array.empty_like(a_dev)
        self.program.substract(self.queue, a_dev.shape, None, a_dev.data, b_dev.data, c_dev.data)
        self.output_layer_error = c_dev.get()
        
        # compute hidden layers error
        for i in range(self.hidden_layers_nbr):
            self.hidden_layers_error.append([])
            self.hidden_layers_error[i] = np.zeros(self.hidden_neurons_nbr)
            for j in range(self.hidden_neurons_nbr):
                output_layer_weight = np.array([], dtype=np.float32)
                for k in range(self.output_nbr):
                    output_layer_weight = np.append(output_layer_weight, self.output_layer_weights[k][j])
                a_dev = cl_array.to_device(self.queue, self.output_layer_error)
                b_dev = cl_array.to_device(self.queue, np.array(output_layer_weight, dtype=np.float32))
                c_dev = cl_array.empty_like(a_dev)
                self.program.multiply(self.queue, a_dev.shape, None, a_dev.data, b_dev.data, c_dev.data)
                sum = c_dev.get()
                
                a_dev = cl_array.to_device(self.queue, sum)
                sum_dev = cl_array.to_device(self.queue, np.array([0], dtype=np.float32))
                self.program.sum(self.queue, a_dev.shape, None, a_dev.data, sum_dev.data)
                self.hidden_layers_error[i][j] = sum_dev.get()[0]
        # compute output layer weights
        for i in range(self.output_nbr):
            a_dev = cl_array.to_device(self.queue, self.hidden_layers[-1])
            b_dev = cl_array.to_device(self.queue, np.array(self.output_layer_error[i], dtype=np.float32))
            c_dev = cl_array.empty_like(a_dev)
            self.program.multiply(self.queue, a_dev.shape, None, a_dev.data, b_dev.data, c_dev.data)
            res = c_dev.get()
            
            a_dev = cl_array.to_device(self.queue, res)
            b_dev = cl_array.to_device(self.queue, np.array([self.LEARNING_RATE], dtype=np.float32))
            c_dev = cl_array.empty_like(a_dev)
            self.program.multiply_by_scalar(self.queue, a_dev.shape, None, a_dev.data, c_dev.data, b_dev.data)
            self.output_layer_weights[i] += c_dev.get()
        # compute hidden layers weights
        for i in range(self.hidden_layers_nbr):
            for j in range(self.hidden_neurons_nbr):
                if i == 0:
                    a_dev = cl_array.to_device(self.queue, input)
                    b_dev = cl_array.to_device(self.queue, np.array(self.hidden_layers_error[i][j], dtype=np.float32))
                    c_dev = cl_array.empty_like(a_dev)
                    self.program.multiply(self.queue, a_dev.shape, None, a_dev.data, b_dev.data, c_dev.data)
                    
                    a_dev = cl_array.to_device(self.queue, c_dev.get())
                    b_dev = cl_array.to_device(self.queue, np.array([self.LEARNING_RATE], dtype=np.float32))
                    c_dev = cl_array.empty_like(a_dev)
                    self.program.multiply_by_scalar(self.queue, a_dev.shape, None, a_dev.data, c_dev.data, b_dev.data)
                    self.hidden_layers_weights[i][j] += c_dev.get()
                else:
                    a_dev = cl_array.to_device(self.queue, self.hidden_layers[i-1])
                    b_dev = cl_array.to_device(self.queue, np.array(self.hidden_layers_error[i][j], dtype=np.float32))
                    c_dev = cl_array.empty_like(a_dev)
                    self.program.multiply(self.queue, a_dev.shape, None, a_dev.data, b_dev.data, c_dev.data)

                    a_dev = cl_array.to_device(self.queue, c_dev.get())
                    b_dev = cl_array.to_device(self.queue, np.array([self.LEARNING_RATE], dtype=np.float32))
                    c_dev = cl_array.empty_like(a_dev)
                    self.program.multiply_by_scalar(self.queue, a_dev.shape, None, a_dev.data, c_dev.data, b_dev.data)
                    self.hidden_layers_weights[i][j] += c_dev.get()
        # compute output layer bias
        a_dev = cl_array.to_device(self.queue, np.array(self.output_layer_error, dtype=np.float32))
        b_dev = cl_array.to_device(self.queue, np.array([self.LEARNING_RATE], dtype=np.float32))
        c_dev = cl_array.empty_like(a_dev)
        self.program.multiply_by_scalar(self.queue, a_dev.shape, None, a_dev.data, c_dev.data, b_dev.data)
        self.output_layer_bias += c_dev.get()
        
        # compute hidden layers bias
        a_dev = cl_array.to_device(self.queue, np.array(self.hidden_layers_error, dtype=np.float32))
        b_dev = cl_array.to_device(self.queue, np.array([self.LEARNING_RATE], dtype=np.float32))
        c_dev = cl_array.empty_like(a_dev)
        self.program.multiply_by_scalar(self.queue, a_dev.shape, None, a_dev.data, c_dev.data, b_dev.data)
        self.hidden_layers_bias += c_dev.get()

    def train_step_gpu(self, input, expected_output):
        self.forward_propagation_gpu(input)
        self.back_propagation_gpu(input, expected_output)
        
    def init_weights(self):
        for i in range(self.hidden_layers_nbr):
            self.hidden_layers_weights.append([])
            for j in range(self.hidden_neurons_nbr):
                self.hidden_layers_weights[i].append([])
                if i == 0:
                    for k in range(self.input_nbr):
                        self.hidden_layers_weights[i][j].append(random.uniform(-1, 1))
                else:
                    for k in range(self.hidden_neurons_nbr if self.hidden_neurons_nbr > self.input_nbr else self.input_nbr):
                        self.hidden_layers_weights[i][j].append(random.uniform(-1, 1))
        for i in range(self.output_nbr):
            self.output_layer_weights.append([])
            for j in range(self.hidden_neurons_nbr):
                self.output_layer_weights[i].append(random.uniform(-1, 1))

    def init_bias(self):
        for i in range(self.hidden_layers_nbr):
            self.hidden_layers_bias.append([])
            for j in range(self.hidden_neurons_nbr):
                self.hidden_layers_bias[i].append(random.uniform(-1, 1))
        for i in range(self.output_nbr):
            self.output_layer_bias.append(random.uniform(-1, 1))

    def init_dropout(self):
        for i in range(self.hidden_layers_nbr):
            self.dropout.append([])
            for j in range(self.hidden_neurons_nbr):
                self.dropout[i].append(random.uniform(0, 1))
    
    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0

    def relu(self, x):
        return max(0, x)

    def relu_derivative(self, x):
        return 1 if x > 0 else 0
    
    def binary_crossentropy(self, y_true, y_pred):
        return -np.sum(np.multiply(y_true, np.log(y_pred)) + np.multiply(1 - y_true, np.log(1 - y_pred)))
    
    def mse(self, y_true, y_pred):
        return np.sum(np.square(np.subtract(y_true, y_pred)))
    
    def forward_propagation(self, input):
        self.hidden_layers = []
        self.output_layer = []
        for i in range(self.hidden_layers_nbr):
            self.hidden_layers.append([])
            if i == 0:
                self.hidden_layers[i] = np.zeros(self.hidden_neurons_nbr)
                for j in range(self.hidden_neurons_nbr):
                    sum = np.multiply(input, self.hidden_layers_weights[i][j])
                    self.hidden_layers[i][j] = np.sum(sum)
            else:
                self.hidden_layers[i] = np.zeros(self.hidden_neurons_nbr)
                for j in range(self.hidden_neurons_nbr):
                    sum = np.multiply(self.hidden_layers[i-1], self.hidden_layers_weights[i][j])
                    self.hidden_layers[i][j] = np.sum(sum)
            self.hidden_layers[i] = [self.sigmoid(x) for x in self.hidden_layers[i]]
            self.hidden_layers[i] = np.add(self.hidden_layers[i], self.hidden_layers_bias[i])
            # for j in range(self.hidden_neurons_nbr):
            #     if random.uniform(0, 1) < self.DROPOUT_RATE:
            #         self.hidden_layers[i][j] = 0

        self.output_layer = np.zeros(self.output_nbr)
        for i in range(self.output_nbr):
            sum = np.multiply(self.hidden_layers[-1], self.output_layer_weights[i])
            self.output_layer[i] = np.sum(sum)
            self.output_layer[i] += self.output_layer_bias[i]
            self.output_layer[i] = self.sigmoid(self.output_layer[i])
            
    def train_step(self, input, expected_output):
        self.forward_propagation(input)
        self.back_propagation(input, expected_output)

    def back_propagation(self, input, expected_output):
        self.output_layer_error = []
        self.hidden_layers_error = []
        
        # compute output layer error
        self.output_layer_error = np.subtract(expected_output, self.output_layer)
        
        # compute hidden layers error
        for i in range(self.hidden_layers_nbr):
            self.hidden_layers_error.append([])
            self.hidden_layers_error[i] = np.zeros(self.hidden_neurons_nbr)
            for j in range(self.hidden_neurons_nbr):
                for k in range(self.output_nbr):
                    self.hidden_layers_error[i][j] += self.output_layer_error[k] * self.output_layer_weights[k][j]
        # compute output layer weights
        for i in range(self.output_nbr):
            res = np.multiply(self.hidden_layers[-1], self.output_layer_error[i])
            res = np.multiply(res, self.LEARNING_RATE)
            self.output_layer_weights[i] += res
        # compute hidden layers weights
        for i in range(self.hidden_layers_nbr):
            for j in range(self.hidden_neurons_nbr):
                if i == 0:
                    res = np.multiply(input, self.hidden_layers_error[i][j])
                    res = np.multiply(res, self.LEARNING_RATE)
                    self.hidden_layers_weights[i][j] += res
                else:
                    res = np.multiply(self.hidden_layers[i-1], self.hidden_layers_error[i][j])
                    res = np.multiply(res, self.LEARNING_RATE)
                    self.hidden_layers_weights[i][j] += res
        # compute output layer bias
        res = np.multiply(self.output_layer_error, self.LEARNING_RATE)
        self.output_layer_bias += res
        res = np.multiply(self.hidden_layers_error, self.LEARNING_RATE)
        self.hidden_layers_bias += res
    
    def predict(self, input):
        self.forward_propagation(input)
        output = self.output_layer
        return output
    
    def format_output(self, output):
        for i in range(len(output)):
            if output[i] > 0.5:
                output[i] = 1
            else:
                output[i] = 0
        output = output.astype(int)
        return output

    def saveModel(self, filename):
        if self.hidden_layers_nbr == 1:
            np.savez(filename,
                hidden_layer_weights=self.hidden_layers_weights,
                output_layer_weights=self.output_layer_weights,
                hidden_layer_bias=self.hidden_layers_bias,
                output_layer_bias=self.output_layer_bias,
                dropout=self.dropout,
                hidden_neurons_nbr=self.hidden_neurons_nbr,
                input_nbr=self.input_nbr,
                output_nbr=self.output_nbr,
                hidden_layers_nbr=self.hidden_layers_nbr,
                accuracy=self.accuracy,
                loss=self.loss,
                val_accuracy=self.val_accuracy,
                val_loss=self.val_loss,
            )
        else:
            weights0 = self.hidden_layers_weights[0]
            hidden_layers_weights = self.hidden_layers_weights[1:]
            bias0 = self.hidden_layers_bias[0]
            hidden_layers_bias = self.hidden_layers_bias[1:]
            np.savez(filename,
                hidden_bias0=bias0,
                hidden_weights0=weights0,
                hidden_layer_weights=hidden_layers_weights,
                output_layer_weights=self.output_layer_weights,
                hidden_layer_bias=hidden_layers_bias,
                output_layer_bias=self.output_layer_bias,
                dropout=self.dropout,
                hidden_neurons_nbr=self.hidden_neurons_nbr,
                input_nbr=self.input_nbr,
                output_nbr=self.output_nbr,
                hidden_layers_nbr=self.hidden_layers_nbr,
                accuracy=self.accuracy,
                loss=self.loss,
                val_accuracy=self.val_accuracy,
                val_loss=self.val_loss,
            )

    def loadModel(self, filename):
        data = np.load(filename)
        self.hidden_layers_nbr = data['hidden_layers_nbr']
        self.hidden_layers_weights = data['hidden_layer_weights']
        self.hidden_layers_bias = data['hidden_layer_bias']
        self.hidden_neurons_nbr = data['hidden_neurons_nbr']
        self.input_nbr = data['input_nbr']
        self.output_nbr = data['output_nbr']
        self.output_layer_weights = data['output_layer_weights']
        self.output_layer_bias = data['output_layer_bias']
        self.dropout = data['dropout']
        self.last_accuracy = data['accuracy']
        self.last_loss = data['loss']
        self.last_val_accuracy = data['val_accuracy']
        self.last_val_loss = data['val_loss']
        
        if self.hidden_layers_nbr > 1:
            weights0 = data['hidden_weights0']
            bias0 = data['hidden_bias0']
            hidden_layers_weights = self.hidden_layers_weights
            hidden_layers_bias = self.hidden_layers_bias
            
            self.hidden_layers_weights = []
            self.hidden_layers_bias = []
            
            self.hidden_layers_weights.append(weights0)
            self.hidden_layers_bias.append(bias0)
            for i in range(len(hidden_layers_weights)):
                self.hidden_layers_weights.append(hidden_layers_weights[i])
            for i in range(len(hidden_layers_bias)):
                self.hidden_layers_bias.append(hidden_layers_bias[i])                
        else:
            self.hidden_layers_weights = self.hidden_layers_weights.tolist()
            self.hidden_layers_bias = self.hidden_layers_bias.tolist()
        self.output_layer_weights = self.output_layer_weights.tolist()
        self.output_layer_bias = self.output_layer_bias.tolist()
        self.dropout = self.dropout.tolist()
        return self
    
    def build_kernel(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, """
            __kernel void multiply(__global const float *a,
                                    __global const float *b,
                                    __global float *c)
            {
                int gid = get_global_id(0);
                c[gid] = a[gid] * b[gid];
            }
            __kernel void add(__global const float *a,
                                    __global const float *b,
                                    __global float *c)
            {
                int gid = get_global_id(0);
                c[gid] = a[gid] + b[gid];
            }
            __kernel void multiply_by_scalar(__global const float *a,
                                    __global float *c,
                                    __global float *scalar)
            {
                int gid = get_global_id(0);
                c[gid] = a[gid] * scalar[0];
            }
            __kernel void sigmoid(__global float *a)
            {
                int gid = get_global_id(0);
                a[gid] = 1 / (1 + exp(-a[gid]));
            }
            __kernel void sum(__global const float *a,
                                    __global float *c)
            {
                int gid = get_global_id(0);
                c[0] += a[gid];
            }
            __kernel void substract(__global const float *a,
                                    __global const float *b,
                                    __global float *c)
            {
                int gid = get_global_id(0);
                c[gid] = a[gid] - b[gid];
            }
            """).build()
    
    def train(self, X, y, epochs=1, X_val=None, y_val=None, gpu=False):
        foldername = '{}inputs_{}hidden_{}neurons/'.format(self.input_nbr, self.hidden_layers_nbr, self.hidden_neurons_nbr)
        if gpu:
            self.build_kernel()
        try:
            rmtree("MODEL/" + foldername)
            mkdir("MODEL/" + '{}inputs_{}hidden_{}neurons'.format(self.input_nbr, self.hidden_layers_nbr, self.hidden_neurons_nbr))
        except:
            mkdir("MODEL/" + '{}inputs_{}hidden_{}neurons'.format(self.input_nbr, self.hidden_layers_nbr, self.hidden_neurons_nbr))
        val_accuracy_temp = []
        val_loss_temp = []
        accuracy_0_1 = []
        loss_training = []
        first_eval = 0
        for j in range(epochs):
            for i in range(len(X)):
                start = time.time()
                if gpu:
                    self.train_step_gpu(X[i], y[i])
                else:
                    self.train_step(X[i], y[i])
                loss_training.append(self.loss_function(y[i], self.output_layer))
                self.loss.append(np.average(loss_training))
                prediction = self.output_layer
                prediction = self.format_output(prediction)
                accuracy_0_1.append(1 if np.array_equal(prediction, y[i]) else 0)
                self.accuracy.append(sum(accuracy_0_1) / len(accuracy_0_1))
                print("\rEpoch:\t{}/{}\t{}/{}\t{}%\tlast step time: {} ms\tloss: {}\taccuracy: {}".format(j+1, epochs, str(i+1).zfill(len(str(len(X)))), len(X), round((i+1) / (len(X)) * 100, 1), str(round((time.time() - start) * 1000, 1)).zfill(5), round(np.average(self.loss), 4), round(self.accuracy[-1], 4)), end='', flush=True)
                if i % 10000 == 0:
                    try:
                        remove("MODEL/" + foldername + 'last{}inputs_{}hidden_{}neurons_{}boards_model.npz'.format(self.input_nbr, self.hidden_layers_nbr, self.hidden_neurons_nbr, i-10000))
                    except:
                        pass
                    self.saveModel("MODEL/" + foldername + 'last{}inputs_{}hidden_{}neurons_{}boards_model.npz'.format(self.input_nbr, self.hidden_layers_nbr, self.hidden_neurons_nbr, i))
                if X_val is not None and y_val is not None:
                    if first_eval == i:
                        first_eval += int(len(X_val) * 0.25)
                        accuracy, loss = self.evaluate(X_val, y_val)
                        val_accuracy_temp.append(accuracy)
                        val_loss_temp.append(loss)
                        self.val_accuracy.append(np.average(val_accuracy_temp))
                        self.val_loss.append(np.average(val_loss_temp))
        accuracy, loss = self.evaluate(X_val, y_val)
        val_accuracy_temp.append(accuracy)
        val_loss_temp.append(loss)
        self.val_accuracy.append(np.average(val_accuracy_temp))
        self.val_loss.append(np.average(val_loss_temp))
        self.plot_loss_accuracy(X_val, y_val, foldername)
            
    def plot_loss_accuracy(self, X_val, y_val, foldername):
        import matplotlib.pyplot as plt
        accuracy = np.interp(np.linspace(0, 1, len(self.accuracy)), np.linspace(0, 1, len(self.accuracy)), self.accuracy)
        plt.plot(accuracy)
        if X_val is not None and y_val is not None:
            self.val_accuracy = np.interp(np.linspace(0, 1, len(self.accuracy)), np.linspace(0, 1, len(self.val_accuracy)), self.val_accuracy)
            plt.plot(self.val_accuracy)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('steps')
        if X_val is not None and y_val is not None:
            plt.legend(['train', 'validation'], loc='upper left')
        plt.title(foldername[:-1] + ' model accuracy')
        plt.gcf().set_size_inches(20, 10)
        plt.savefig("MODEL/" + foldername + 'last_training_accuracy.png')
        plt.clf()
        loss = np.interp(np.linspace(0, 1, len(self.loss)), np.linspace(0, 1, len(self.loss)), self.loss)
        plt.plot(loss)
        if X_val is not None and y_val is not None:
            self.val_loss = np.interp(np.linspace(0, 1, len(self.loss)), np.linspace(0, 1, len(self.val_loss)), self.val_loss)
            plt.plot(self.val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('steps')
        if X_val is not None and y_val is not None:
            plt.legend(['train', 'validation'], loc='upper left')
        plt.title(foldername[:-1] + ' model loss')
        plt.gcf().set_size_inches(20, 10)
        plt.savefig("MODEL/" + foldername + 'last_training_loss.png')
        plt.clf()
        
    def evaluate_single(self, X, y):
        prediction = self.predict(X)
        loss = np.sum(np.square(np.subtract(y, prediction)))
        for i in range(len(prediction)):
            if prediction[i] > 0.5:
                prediction[i] = 1
            else:
                prediction[i] = 0
        if np.array_equal(prediction, y):
            return 1, loss
        return 0, loss

    def evaluate(self, X, y):
        pool = mp.Pool(mp.cpu_count() // 4)
        results = pool.starmap(self.evaluate_single, zip(X, y))
        pool.close()
        pool.join()
        return sum([x[0] for x in results]) / len(results), sum([x[1] for x in results]) / len(results)