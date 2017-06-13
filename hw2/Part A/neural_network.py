from random import uniform
from math import exp
from os import path
from pickle import dump, HIGHEST_PROTOCOL


BOTTOM_RANGE = -0.3
UPPER_RANGE = 0.3


def neuron_activation(neuron_value):
    return 1.0 / (1.0 + exp(-neuron_value))


def neuron_activation_derivative(activation_output):
    return activation_output * (1.0 - activation_output)


class Neuron(object):
    def __init__(self, number_of_neurons_in_previous_layer):
        self.value = 0
        self.error = 0
        self.weights = [uniform(BOTTOM_RANGE, UPPER_RANGE) for _ in range(number_of_neurons_in_previous_layer)]

    def get_value(self):
        return self.value

    def get_error(self):
        return self.error

    def calculate_neuron_value(self, previous_layer):
        input_weights_sum = 0
        for index, weight in enumerate(self.weights):
            input_weights_sum += previous_layer[index].get_value() * weight
        self.value = neuron_activation(input_weights_sum)

    def calculate_error_by_neurons_layer(self, neuron_index, next_layer):
        error_weight_sum = 0
        for neuron in next_layer:
            error_weight_sum += neuron.get_error() * neuron.weights[neuron_index]
        self.error = error_weight_sum * neuron_activation_derivative(self.get_value())

    def update_neuron_weights(self, previous_layer, network_learning_rate):
        for index, weight in enumerate(self.weights):
            self.weights[index] += network_learning_rate * self.get_error() * previous_layer[index].get_value()


class Network(object):
    def __init__(self, number_of_neurons_in_hidden_layer, output_size, learning_rate, training_samples):
        self.input_layer = [Neuron(input_size) for _ in range(training_samples)]
        self._initialize_input_layer_neurons_value(training_samples)
        input_layer_bias_neuron = Neuron(0)
        input_layer_bias_neuron.value = 1
        self.input_layer.append(input_layer_bias_neuron)
        input_size = len(training_samples)
        self.hidden_layer = [Neuron(input_size) for _ in range(number_of_neurons_in_hidden_layer)]
        hidden_layer_bias_neuron = Neuron(input_size)
        hidden_layer_bias_neuron.value = 1
        self.hidden_layer.append(hidden_layer_bias_neuron)
        self.output_layer = [Neuron(len(self.hidden_layer)) for _ in range(output_size)]
        self.__epochs = 1
        self.__learning_rate = learning_rate
        self.__training_samples = training_samples
        self.__error_rate = None

    def __calculate_net_output(self):
        for neuron in self.hidden_layer[:-1]:
            neuron.calculate_neuron_value(self.input_layer)
        for neuron in self.output_layer:
            neuron.calculate_neuron_value(self.hidden_layer)
        return [output_neuron.get_value() for output_neuron in self.output_layer]

    def __calculate_neurons_error(self, expected_values):
        next_layer = self.output_layer
        for neuron, expected_value in zip(self.output_layer, expected_values):
            neuron.error = (expected_value - neuron.get_value()) * neuron_activation_derivative(neuron.get_value())
        for neuron_index, neuron in enumerate(self.hidden_layer):
            neuron.calculate_error_by_neurons_layer(neuron_index, next_layer)

    def __update_neurons_weights(self, network_learning_rate):
        for neuron in self.hidden_layer:
            neuron.update_neuron_weights(self.input_layer, network_learning_rate)
        for neuron in self.output_layer:
            neuron.update_neuron_weights(self.hidden_layer, network_learning_rate)

    def training_neurons_network(self):
        self.__epochs = 1
        stop_network_learning = False
        last_error_rate = 0
        while stop_network_learning is False:
            error_rate = 0
            for train_input in self.__training_samples:
                expected_output_values = train_input[:]
                for input_neuron_index, input_field in zip(range(len(self.input_layer)), training_samples):
                    self.input_layer[input_neuron_index].value = input_field
                output_values = self.__calculate_net_output()
                for output_value, expected_output_value in zip(output_values, expected_output_values):
                    error_rate += (expected_output_value - output_value) ** 2
                self.__calculate_neurons_error(expected_output_values)
                self.__update_neurons_weights(self.__learning_rate)
            is_first_learning_iteration = self.__epochs == 1
            if is_first_learning_iteration is False:
                stop_network_learning = last_error_rate < error_rate
            last_error_rate = error_rate
            self.__epochs += 1
        self.__error_rate = last_error_rate

    def dump_nn_to_pickle(self, model_num, folder_path):
        filename = "model_num_-{}-_hidden_size_-{}-_learning_rate_-{}"\
            .format(str(model_num), str(len(self.hidden_layer)), str(self.__learning_rate)) + '.txt'
        with open(path.join(folder_path, filename), 'wb') as model_file:
            dump(self, model_file, HIGHEST_PROTOCOL)

    def __str__(self):
        print("BlaBlaBla {}".format(self.__learning_rate))
