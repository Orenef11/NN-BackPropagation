# coding=utf-8
from itertools import product
from random import uniform
import matplotlib.pylab as plt
from os import makedirs, path
import logging

NUMBER_OF_PLOT_DIMENSIONS = 2
MAXIMUM_NUMBER_OF_ITERATIONS = 32


class Neuron(object):
    def __init__(self):
        self._neuron_location_parameters = []
        self._neighbours = []
        self._initialize_neuron_location()

    def _initialize_neuron_location(self):
        for _ in range(NUMBER_OF_PLOT_DIMENSIONS):
            self._neuron_location_parameters.append(uniform(0.49, 0.51))  # temporary values choice!

    def get_neuron_neighbours(self):
        return self._neighbours

    def update_neuron_neighbours_collection(self, neighbour_neuron):
        self._neighbours.append(neighbour_neuron)

    def draw_neuron(self):
        plt.title('Neurons Connection')
        plt.plot(self._neuron_location_parameters[0], self._neuron_location_parameters[1],
                              "ro", color="blue")

    def calculate_distance_from_data_point(self, data_point):
        raw_distance_value = 0
        for first_point_parameter, second_point_parameter in zip(data_point,  self.get_neuron_location()):
            raw_distance_value += (first_point_parameter - second_point_parameter) ** 2
        return raw_distance_value ** 0.5

    def update_neuron_location(self, data_point, learning_rate):
        for parameter_index in range(len(self._neuron_location_parameters)):
            diffrence_from_data_point = data_point[parameter_index] - self._neuron_location_parameters[parameter_index]
            self._neuron_location_parameters[parameter_index] += learning_rate * diffrence_from_data_point

    def get_neuron_location(self):
        return self._neuron_location_parameters



class KohonenNetwork(object):
    def __init__(self, parameters):
        network_dimensions_tuple, dataset, learning_rate, model_name, images_output_folder_path = \
            parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
        self.__model_name = model_name
        self.__dataset = dataset
        self._network_dimensions = network_dimensions_tuple
        self._learning_rate = learning_rate
        self._number_of_neurons = self._calculate_number_of_network_neurons()
        self._network_neurons = {}
        self.__images_folder_path = path.join(images_output_folder_path, 'model ' + str(model_name))
        self._initialize_network_neurons()
        self.__images_count = 1
        for (neuron_indexes, neuron) in self._network_neurons.items():
            self._add_neuron_neighbours(neuron_indexes, neuron)

        if not path.isdir(self.__images_folder_path):
            makedirs(self.__images_folder_path)

        self.train_network()

    def _add_neuron_neighbours(self, neuron_indexes, neuron):
        number_of_indexes = len(neuron_indexes)
        for increased_index_place in range(number_of_indexes):
            searched_indexes = list(neuron_indexes[:])
            searched_indexes[increased_index_place] += 1
            search_result = self._network_neurons.get(tuple(searched_indexes), None)
            if search_result is not None:
                search_result.update_neuron_neighbours_collection(neuron)
                neuron.update_neuron_neighbours_collection(search_result)

    def _initialize_network_neurons(self):
        neurons_indexes_by_network_dimensions = []
        for network_dimension in self._network_dimensions:
            neurons_indexes_by_network_dimensions.append(range(network_dimension))
        for neurons_indexes in product(*neurons_indexes_by_network_dimensions):
            self._network_neurons.update({neurons_indexes: Neuron()})

    def _calculate_number_of_network_neurons(self):
        number_of_neurons = 1
        for network_dimension in self._network_dimensions:
            number_of_neurons *= network_dimension
        return number_of_neurons

    def _draw_neurons(self):
        for neuron in self._network_neurons.values():
            neuron.draw_neuron()

    def _draw_neurons_connections(self):
        for neuron in self._network_neurons.values():
            for neuron_neighbour in neuron.get_neuron_neighbours():
                neuron_location = neuron.get_neuron_location()
                neuron_neighbour_location = neuron_neighbour.get_neuron_location()
                plt.plot([neuron_location[0], neuron_neighbour_location[0]],
                         [neuron_location[1], neuron_neighbour_location[1]], color="green")

    def draw_network(self, save_image_flag=False):
        if self.__images_count != 1:
            plt.axis(self.__dataset.axis_range)
        else:
            self.__dataset.draw_data(True, path.join(self.__images_folder_path, 'dataset.png'))
        title = '"Neurons Connection\nl_rate={}, dataset_type={}'.format(self._learning_rate,
                                                                         self.__dataset.dataset_type)
        if type(self._network_dimensions) is tuple:
            title += 'model_dimensions={}, {}'.format(self._network_dimensions[0], self._network_dimensions[1])
        elif type(self._network_dimensions) is int:
            title += 'model_dimensions={}'.format(self._network_dimensions)

        self._draw_neurons()
        self._draw_neurons_connections()
        plt.title(title)
        if save_image_flag:
            plt.savefig(path.join(self.__images_folder_path, str(self.__images_count) + '.png'))
        else:
            plt.show()
        self.__images_count += 1
        plt.clf()

    def _get_neuron_with_smallest_distance_from_data_point(self, data_point):
        neurons = list(self._network_neurons.values())
        minimum_distance_neuron = neurons[0]
        minimal_distance = neurons[0].calculate_distance_from_data_point(data_point)
        for neuron in neurons[1:]:
            neuron_distance_from_data_point = neuron.calculate_distance_from_data_point(data_point)
            if neuron_distance_from_data_point < minimal_distance:
                minimal_distance = neuron_distance_from_data_point
                minimum_distance_neuron = neuron
        return minimum_distance_neuron

    def train_network(self):
        for number_of_iteration in range(MAXIMUM_NUMBER_OF_ITERATIONS):
            if number_of_iteration % 5 == 0:
                self.draw_network(True)
                logging.debug("\nIteration number: {}".format(number_of_iteration))
            for data_point in self.__dataset.get_data_points():
                minimum_distance_neuron = self._get_neuron_with_smallest_distance_from_data_point(data_point)
                minimum_distance_neuron.update_neuron_location(data_point, self._learning_rate)
                for neighbour_neuron in minimum_distance_neuron.get_neuron_neighbours():
                    neighbour_neuron.update_neuron_location(data_point, self._learning_rate)
