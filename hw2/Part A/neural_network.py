from random import random


class Neuron:
    def __init__(self, number_of_neurons_in_previous_layer):
        self.value = 0
        self.error = 0
        self.weights = [random() for _ in range(number_of_neurons_in_previous_layer)]

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


class Network:
    def __init__(self, input_size, output_size):
        self.output_layer = []
        self.hidden_layer = [Neuron(input_size) for _ in range(number_of_neurons_in_hidden_layer)]
        bias_neuron = Neuron(input_size)
        bias_neuron.value = 1
        self.hidden_layer.append(bias_neuron)
        self.output_layer = [Neuron(len(self.hidden_layer)) for _ in range(output_size)]

    def calculate_net_output(self, input_neurons):
        for neuron in self.hidden_layer[:-1]:
            neuron.calculate_neuron_value(input_neurons)
        for neuron in self.output_layer:
            neuron.calculate_neuron_value(self.hidden_layer)
        return [output_neuron.get_value() for output_neuron in self.output_layer]

    def calculate_neurons_error(self, expected_values):
        next_layer = self.output_layer
        for neuron, expected_value in zip(self.output_layer, expected_values):
            neuron.error = (expected_value - neuron.get_value()) * neuron_activation_derivative(neuron.get_value())
        for neuron_index, neuron in enumerate(self.hidden_layer):
            neuron.calculate_error_by_neurons_layer(neuron_index, next_layer)

    def update_neurons_weights(self, input_neurons, network_learning_rate):
        for neuron in self.hidden_layer:
            neuron.update_neuron_weights(input_neurons, network_learning_rate)
        for neuron in self.output_layer:
            neuron.update_neuron_weights(self.hidden_layer, network_learning_rate)
