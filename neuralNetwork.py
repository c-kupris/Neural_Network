# Make a neural network!
import keras.layers
import matplotlib.pyplot
import numpy.ma
import tensorflow


class Network:
    # Make a class for the network

    # Define weights

    weight_one = float
    weight_two = float
    weight_three = float
    weight_four = float
    weight_five = float
    weight_six = float
    weight_seven = float
    weight_eight = float
    weight_nine = float
    weight_ten = float

    # Define biases

    bias_one = float
    bias_two = float
    bias_three = float
    bias_four = float
    bias_five = float
    bias_six = float
    bias_seven = float
    bias_eight = float
    bias_nine = float
    bias_ten = float

    # Ground Truth

    true_val = int()
    predicted_val = int()

    # Compute error to see how effective the network is

    error = ((predicted_val - true_val) ^ 2) / true_val

    # Loss function - function to be optimized

    @classmethod
    def loss_function(cls):
        loss = tensorflow.keras.losses.CategoricalCrossentropy
        return loss

    # Flatten input images using numpy 'flatten' method

    flattened_inputs = numpy.ma.flatten_mask(mask=[0, 0, 1])


    # Preprocessing layer

    preprocessing_layer = keras.layers.PreprocessingLayer
    preprocessing_layer.trainable = True
    preprocessing_layer.add_weight = weight_one, weight_eight, weight_nine, weight_ten, weight_seven, weight_three, weight_five, weight_four
    preprocessing_layer.adapt(keras.layers.PreprocessingLayer(), data="image", batch_size=50, steps=10)
    preprocessing_layer.compile(self=keras.layers.PreprocessingLayer())

    # Layer One
    layer_one = keras.layers.InputLayer
    layer_one.trainable = True
    layer_one.add_weight = weight_one, weight_two, weight_three
    layer_one.add_loss(self=layer_one.Layer(), losses=loss_function)

    # Hidden layer one
    hidden_layer_one = keras.layers.ReLU
    hidden_layer_one.add_loss(keras.layers.Layer(), loss_function)
    hidden_layer_one.trainable = True
    hidden_layer_one.add_weight = weight_one, weight_four, weight_five, weight_six, weight_seven

    # Layer two
    layer_two = keras.layers.Layer()
    layer_two.add_loss(losses=loss_function())
    layer_two.add_weight = weight_eight, weight_six, weight_two
    layer_two.trainable = True

    # Hidden Layer two
    hidden_layer_two = keras.layers.activation.softmax
    hidden_layer_two.Layer.trainable = True
    hidden_layer_two.Layer.add_weight = weight_one, weight_eight
    hidden_layer_two.Layer.add_loss(self=keras.layers.Layer(), losses=loss_function())

    # Output layer
    output_layer = keras.layers.Layer()
    output_layer.add_loss = loss_function()
    output_layer.add_weight = weight_five, weight_six, weight_seven
    output_layer.trainable = True
    output_layer.activity_regularizer = keras.layers.activation.softmax

    # Modelling

    model = keras.models.clone_and_build_model

    # Graph the error as the training increases

    graph = matplotlib.pyplot.plot(model, error)
    