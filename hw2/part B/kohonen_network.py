class Neuron(object):
    pass


class KohonenNetwork(object):
    def __init__(self, network_dimensions):
        self._network_dimensions = network_dimensions
        self._number_of_neurons = self._calculate_number_of_network_neurons()
        self._network_neurons = [Neuron() for _ in range(self._number_of_neurons)]

    def _calculate_number_of_network_neurons(self):
        return reduce(lambda x, y: x * y, self._network_dimensions)
