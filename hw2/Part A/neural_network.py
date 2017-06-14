from random import uniform
from os import path, makedirs
from pickle import dump, HIGHEST_PROTOCOL

BOTTOM_RANGE = -0.4
UPPER_RANGE = 0.4
MODELS_FOLDER = 'Models structure'


class Neuron(object):
    def __init__(self, number_of_neurons_in_previous_layer):
        self.value = 0
        self.error = 0
        self.weights = [uniform(BOTTOM_RANGE, UPPER_RANGE) for _ in range(number_of_neurons_in_previous_layer)]

    def get_value(self):
        return self.value

    def get_error(self):
        return self.error

    def calculate_neuron_value(self, previous_layer, neuron_activiation_func):
        input_weights_sum = 0
        for index, weight in enumerate(self.weights):
            input_weights_sum += previous_layer[index].get_value() * weight
        self.value = neuron_activiation_func(input_weights_sum)

    def calculate_error_by_neurons_layer(self, neuron_index, next_layer, neuron_activiation_func_derivative):
        error_weight_sum = 0
        for neuron in next_layer:
            error_weight_sum += neuron.get_error() * neuron.weights[neuron_index]
        self.error = error_weight_sum * neuron_activiation_func_derivative(self.get_value())

    def update_neuron_weights(self, previous_layer, network_learning_rate):
        for index, weight in enumerate(self.weights):
            self.weights[index] += network_learning_rate * self.get_error() * previous_layer[index].get_value()


class Network(object):
    # Because we use multiple cores, we could not move all the variables.
    # Therefore, the passed variable contains all the following variables:
    # number_of_neurons_in_hidden_layer, output_size, learning_rate, image_convert_obj,
    #              neurons_activation_function, neurons_ativiation_function_derivative
    def __init__(self, parameters):
        number_of_neurons_in_hidden_layer, output_size, learning_rate, image_convert_obj, neurons_activation_function, \
            neurons_ativiation_function_derivative, filename = \
            parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6]
        self.__filename = str(filename)
        self.image_convert_obj = image_convert_obj
        training_samples = image_convert_obj.read_sub_images_file(image_convert_obj.image_regular_des_path)
        self.input_layer = [Neuron(0) for _ in range(len(training_samples))]
        input_layer_bias_neuron = Neuron(0)
        input_layer_bias_neuron.value = 1
        self.input_layer.append(input_layer_bias_neuron)
        input_size = len(training_samples)
        self.hidden_layer = [Neuron(input_size) for _ in range(number_of_neurons_in_hidden_layer)]
        hidden_layer_bias_neuron = Neuron(input_size)
        hidden_layer_bias_neuron.value = 1
        self.hidden_layer.append(hidden_layer_bias_neuron)
        self.output_layer = [Neuron(len(self.hidden_layer)) for _ in range(output_size)]
        self._neurons_activation_function = neurons_activation_function
        self._neurons_ativiation_function_derivative = neurons_ativiation_function_derivative
        self.__epochs = 1
        self.__learning_rate = learning_rate
        self.__training_samples = training_samples
        self.__error_rate = None

        self.training_neurons_network()

    def __calculate_net_output(self):
        for neuron in self.hidden_layer[:-1]:
            neuron.calculate_neuron_value(self.input_layer, self._neurons_activation_function)
        for neuron in self.output_layer:
            neuron.calculate_neuron_value(self.hidden_layer, self._neurons_activation_function)
        return [output_neuron.get_value() for output_neuron in self.output_layer]

    def __calculate_neurons_error(self, expected_values):
        next_layer = self.output_layer
        for neuron, expected_value in zip(self.output_layer, expected_values):
            neuron.error = (expected_value - neuron.get_value()) * self._neurons_ativiation_function_derivative(
                neuron.get_value())
        for neuron_index, neuron in enumerate(self.hidden_layer):
            neuron.calculate_error_by_neurons_layer(neuron_index, next_layer,
                                                    self._neurons_ativiation_function_derivative)

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
                for input_neuron_index, sub_training_list in zip(range(len(self.input_layer)), self.__training_samples):
                    for pixel_value in sub_training_list:
                        self.input_layer[input_neuron_index].value = pixel_value
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
        self.dump_nn_to_pickle()
        print(self.__filename, self.__error_rate)

    def dump_nn_to_pickle(self):
        if not path.isdir(MODELS_FOLDER):
            makedirs(MODELS_FOLDER)
        filename = "model_num_-{}-_hidden_size_-{}-_learning_rate_-{}"\
                   .format(self.__filename, str(len(self.hidden_layer)), str(self.__learning_rate)) + '.pickle'
        with open(path.join(MODELS_FOLDER, filename), 'wb') as model_file:
            dump(self, model_file, HIGHEST_PROTOCOL)
