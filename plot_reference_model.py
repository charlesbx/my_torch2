import src.NeuralNetwork as nn
import src.TrainingSet as ts
import matplotlib.pyplot as plt

def plot_models(models):
    for model in models:
        plt.plot(model['model'].last_val_loss, color=model['color1'], label=model['legend1'])
        plt.plot(model['model'].last_loss, color=model['color2'], label=model['legend2'])
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Loss')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    name = models[0]['name']
    nbrref = name.split('_')[1]
    nbr_layers = name.split('_')[-1].split('.')[0]
    # print(nbrref+ '_' + nbr_layers + '_loss.png')
    plt.savefig(nbrref+ '_' + nbr_layers + '_loss.png')
    # plt.show()

################ 1 LAYER ################

model_1920_1 = nn.NeuralNetwork(filename='model_25k-reference_1920_1layer.npz')
model_2048_1 = nn.NeuralNetwork(filename='model_25k-reference_2048_1layer.npz')
model_3072_1 = nn.NeuralNetwork(filename='model_25k-reference_3072_1layer.npz')
model_3840_1 = nn.NeuralNetwork(filename='model_25k-reference_3840_1layer.npz')

model_1920_1 = {
    "name": 'model_25k-reference_1920_1layer.npz',
    "color1": "blue",
    "color2": "orange",
    "legend1": "train 1920",
    "legend2": "val 1920",
    "model": model_1920_1
}
model_2048_1 = {
    "name": 'model_25k-reference_2048_1layer.npz',
    "color1": "green",
    "color2": "red",
    "legend1": "train 2048",
    "legend2": "val 2048",
    "model": model_2048_1
}
model_3072_1 = {
    "name": 'model_25k-reference_3072_1layer.npz',
    "color1": "purple",
    "color2": "brown",
    "legend1": "train 3072",
    "legend2": "val 3072",
    "model": model_3072_1
}
model_3840_1 = {
    "name": 'model_25k-reference_3840_1layer.npz',
    "color1": "black",
    "color2": "yellow",
    "legend1": "train 3840",
    "legend2": "val 3840",
    "model": model_3840_1
}

models = [model_1920_1, model_2048_1, model_3072_1, model_3840_1]
plot_models(models)

################ 2 LAYERS ################

# model_1920_2 = nn.NeuralNetwork(filename='model_25k-reference_1920_2layers.npz')
# model_2048_2 = nn.NeuralNetwork(filename='model_25k-reference_2048_2layers.npz')
# model_3072_2 = nn.NeuralNetwork(filename='model_25k-reference_3072_2layers.npz')
# model_3840_2 = nn.NeuralNetwork(filename='model_25k-reference_3840_2layers.npz')

# model_1920_2 = {
#     "name": 'model_25k-reference_1920_2layers.npz',
#     "color1": "blue",
#     "color2": "orange",
#     "legend1": "train 1920",
#     "legend2": "val 1920",
#     "model": model_1920_2
# }
# model_2048_2 = {
#     "name": 'model_25k-reference_2048_2layers.npz',
#     "color1": "green",
#     "color2": "red",
#     "legend1": "train 2048",
#     "legend2": "val 2048",
#     "model": model_2048_2
# }
# model_3072_2 = {
#     "name": 'model_25k-reference_3072_2layers.npz',
#     "color1": "purple",
#     "color2": "brown",
#     "legend1": "train 3072",
#     "legend2": "val 3072",
#     "model": model_3072_2
# }
# model_3840_2 = {
#     "name": 'model_25k-reference_3840_2layers.npz',
#     "color1": "black",
#     "color2": "yellow",
#     "legend1": "train 3840",
#     "legend2": "val 3840",
#     "model": model_3840_2
# }

# models = [model_1920_2, model_2048_2, model_3072_2, model_3840_2]
# plot_models(models)

############## EXOTIC MODELS ##############

model_7680_1 = nn.NeuralNetwork(filename='model_25k-reference_7680_1layer.npz')
model_7680_2 = nn.NeuralNetwork(filename='model_25k-reference_7680_2layers.npz')
model_768_2 = nn.NeuralNetwork(filename='model_25k-reference_768_2layers.npz')
model_2048_3 = nn.NeuralNetwork(filename='model_25k-reference_2048_3layers.npz')

model_7680_1 = {
    "name": 'model_25k-reference_7680_1layer.npz',
    "color1": "blue",
    "color2": "orange",
    "legend1": "train 7680",
    "legend2": "val 7680",
    "model": model_7680_1
}
model_7680_2 = {
    "name": 'model_25k-reference_7680_2layers.npz',
    "color1": "green",
    "color2": "red",
    "legend1": "train 7680",
    "legend2": "val 7680",
    "model": model_7680_2
}
model_768_2 = {
    "name": 'model_25k-reference_768_2layers.npz',
    "color1": "purple",
    "color2": "brown",
    "legend1": "train 768",
    "legend2": "val 768",
    "model": model_768_2
}
model_2048_3 = {
    "name": 'model_25k-reference_2048_3layers.npz',
    "color1": "black",
    "color2": "yellow",
    "legend1": "train 2048",
    "legend2": "val 2048",
    "model": model_2048_3
}

models = [model_7680_1, model_7680_2, model_768_2, model_2048_3]
plot_models(models)