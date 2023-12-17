from sys import stderr, exit

class Args:
    def __init__(self, argv):
        try:
            self.argv = argv
            self.args = {
                'new': False,
                'load': False,
                'train': False,
                'predict': False,
                'save': False,
                'load_filename': None,
                'dataset_filename': None,
                'save_filename': None,
                'epochs': None,
                'in_layer_neurons_nbr': None,
                'hidden_layers_neurons_nbr': None,
                'out_layer_neurons_nbr': None,
                'hidden_layers_nbr': None,
                'help': False,
                'gpu': False
            }
            self.parse()
            self.check_inconsistencies()
            self.check_values()
        except Exception as e:
            print(e, file=stderr)
            exit(84)
            
    def parse(self):
        # parse the command line arguments
        if len(self.argv) == 1:
            self.help()
            exit(84)
        args = self.argv[1:]
        i = 0
        while i in range(len(args)):
            if args[i] == '-h' or args[i] == '--help':
                self.args['help'] = True
            elif args[i] == '-n' or args[i] == '--new':
                self.args['new'] = True
                # count the number of number that follows -n or --new
                layers = []
                for j in range(i+1, len(args)):
                    try:
                        int(args[j])
                        layers.append(int(args[j]))
                    except:
                        break
                if len(layers) < 3:
                    raise Exception('Invalid argument: -n or --new must be followed by at least 3 integers')
                if len(layers) == 3:
                    self.args['in_layer_neurons_nbr'] = layers[0]
                    self.args['out_layer_neurons_nbr'] = layers[1]
                    self.args['hidden_layers_neurons_nbr'] = layers[2]
                    self.args['hidden_layers_nbr'] = 1
                elif len(layers) == 4:
                    self.args['in_layer_neurons_nbr'] = layers[0]
                    self.args['out_layer_neurons_nbr'] = layers[1]
                    self.args['hidden_layers_neurons_nbr'] = layers[2]
                    self.args['hidden_layers_nbr'] = layers[3]
                i += len(layers)
            elif args[i] == '-l' or args[i] == '--load':
                self.args['load'] = True
                if i+1 >= len(args):
                    raise Exception('Invalid argument: -l or --load must be followed by a filename')
                self.args['load_filename'] = args[i+1]
                i += 1
            elif args[i] == '-t' or args[i] == '--train':
                self.args['train'] = True
                try:
                    int(args[i+1])
                    self.args['epochs'] = int(args[i+1])
                    i += 1
                except:
                    pass
            elif args[i] == '-g' or args[i] == '--gpu':
                self.args['gpu'] = True
            elif args[i] == '-p' or args[i] == '--predict':
                self.args['predict'] = True
            elif args[i] == '-s' or args[i] == '--save':
                self.args['save'] = True
                if i+1 >= len(args):
                    raise Exception('Invalid argument: -s or --save must be followed by a filename')
                self.args['save_filename'] = args[i+1]
                i += 1
            elif self.args['dataset_filename'] == None:
                self.args['dataset_filename'] = args[i]
            else:
                raise Exception('Invalid argument: ' + args[i])
            i += 1
    
    def check_inconsistencies(self):
        if self.args['new'] == True and self.args['load'] == True:
            raise Exception('Invalid argument: -n or --new and -l or --load are mutually exclusive')
        if self.args['new'] == True and self.args['predict'] == True:
            raise Exception('Invalid argument: -n or --new and -p or --predict are mutually exclusive')
        if self.args['train'] == True and self.args['predict'] == True:
            raise Exception('Invalid argument: -t or --train and -p or --predict are mutually exclusive')
        if self.args['load'] == True and (self.args['predict'] == False and self.args['train'] == False):
            raise Exception('Invalid argument: -l or --load must be followed by -p or -t')
        if self.args['new'] == True and (self.args['predict'] == True):
            raise Exception('Invalid argument: -n or --new can only be used with -t or --train')
        if self.args['new'] == False and self.args['load'] == False and self.args['help'] == False:
            raise Exception('Invalid argument: either -n or --new or -l or --load must be present')
        if self.args['dataset_filename'] == None and (self.args['predict'] == True or self.args['train'] == True):
            raise Exception('Invalid argument: FILENAME must be present')
        if self.args['new'] == True and (self.args['in_layer_neurons_nbr'] == None or self.args['out_layer_neurons_nbr'] == None):
            raise Exception('Invalid argument: -n or --new must be followed by IN_LAYER_NEURONS_NBR, HIDDEN_LAYER_NEURONS_NBR and OUT_LAYER_NEURONS_NBR')
        if self.args['load'] == True and self.args['load_filename'] == None:
            raise Exception('Invalid argument: -l or --load must be followed by LOAD_FILENAME')
        if self.args['save'] == True and self.args['save_filename'] == None:
            raise Exception('Invalid argument: -s or --save must be followed by SAVEFILE')
        if self.args['gpu'] == True and self.args['new'] == True:
            raise Exception('Invalid argument: -g or --gpu should be used with -l or --load')
        if self.args['new'] == False and (self.args['in_layer_neurons_nbr'] != None or self.args['out_layer_neurons_nbr'] != None or self.args['hidden_layers_neurons_nbr'] != None or self.args['hidden_layers_nbr'] != None):
            raise Exception('Invalid argument: IN_LAYER_NEURONS_NBR, HIDDEN_LAYER_NEURONS_NBR and OUT_LAYER_NEURONS_NBR can only be used with -n or --new')

    def check_values(self):
        # check the values of the arguments
        if self.args['dataset_filename'] != None:
            filename = self.args['dataset_filename']
            try:
                with open(filename, 'r') as f:
                    pass
            except:
                raise Exception('Invalid argument: FILENAME must be a valid file')
        if self.args['load_filename'] != None:
            load_filename = self.args['load_filename']
            try:
                with open(load_filename, 'r') as f:
                    pass
            except:
                raise Exception('Invalid argument: LOAD_FILENAME must be a valid file')
        if self.args['epochs'] != None:
            epochs = self.args['epochs']
            if epochs < 1:
                raise Exception('Invalid argument: EPOCHS must be a positive integer')
        if self.args['epochs'] == None:
            self.args['epochs'] = 1
        if self.args['in_layer_neurons_nbr'] != None:
            in_layer_neurons_nbr = self.args['in_layer_neurons_nbr']
            if in_layer_neurons_nbr < 1:
                raise Exception('Invalid argument: IN_LAYER_NEURONS_NBR must be a positive integer')
        if self.args['hidden_layers_neurons_nbr'] != None:
            hidden_layers_neurons_nbr = self.args['hidden_layers_neurons_nbr']
            if hidden_layers_neurons_nbr < 1:
                raise Exception('Invalid argument: HIDDEN_LAYER_NEURONS_NBR must be a positive integer')
            hidden_layers_nbr = self.args['hidden_layers_nbr']
            if hidden_layers_nbr < 1:
                raise Exception('Invalid argument: HIDDEN_LAYERS_NBR must be a positive integer')
        if self.args['out_layer_neurons_nbr'] != None:
            out_layer_neurons_nbr = self.args['out_layer_neurons_nbr']
            if out_layer_neurons_nbr < 1:
                raise Exception('Invalid argument: OUT_LAYER_NEURONS_NBR must be a positive integer')
        if self.args['help'] == True:
            self.help()
            exit(0)
            
    def help(self):
        print("USAGE\n\t./my_torch [--new IN_LAYER HIDDEN_LAYERS OUT_LAYER HIDDEN_LAYERS_NBR | --load LOADFILE] [--train | --predict] [--save SAVEFILE] FILE\n")
        print("DESCRIPTION\n\t--new\t\tCreate a new neural network with random weights.\nEach subsequent number represent the number of neurons on each layer, from left to right. For example, ./my_torch --new 3 4 5 will create a neural network with an input layer of 3 neurons, a hidden layer of 4 neurons and an output layer of 5 neurons. You can have as many hidden layers as you want, if you want 3 hidden layers, you can do ./my_torch --new 3 4 5 6 7. You can also have no hidden layers.\n")
        print("\t--load\t\tLoads an existing neural network from LOADFILE.")
        print("\t--train\t\tLaunches the neural network in training mode. Each board in FILE must contain inputs to send to the neural network, as well as the expected output.(optionnaly you can specify the number of epochs to train)")
        print("\t--predict\tLaunches the neural network in predictin mode. Each board in FILE must contain inputs to send to the neural network.")
        print("\t--save\t\tSaves the neural network internal state to SAVEFILE.\n")
        print("\tFILE\t\tFILE containing chessboards")
