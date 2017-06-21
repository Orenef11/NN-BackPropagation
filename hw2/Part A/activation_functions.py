from math import exp


class Activation_Function_Interface(object):
    def __init__(function_type_name,
                 activation_function,
                 activation_function_derivative):
        self.type_name = function_type_name
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def display_function_type_name():
        print "The function type is " + self.type_name


def sigmoid_activation_function(self, neuron_value):
    try:
        return 1.0 / (1.0 + exp(-neuron_value))
    except:
        return -1


def sigmoid_activation_function_derivative(self, activation_output):
    return activation_output * (1.0 - activation_output)


def bipolar_sigmoid_activation_function(neuron_value):
    try:
        return (1.0 - exp(-neuron_value)) / (1.0 + exp(-neuron_value))
    except:
        return -1


def bipolar_sigmoid_activation_function_derivative(activation_output):
        return 0.5 * (1.0 + activation_output) * (1.0 - activation_output)


def hyperbolic_tangent_activation_function(neuron_value):
    try:
        return (1.0 - exp(-2 * neuron_value)) / (1.0 + exp(-2 * neuron_value))
    except:
        return -1


def hyperbolic_tangent_activation_function_derivative(activation_output):
    return (1.0 + activation_output) * (1.0 - activation_output)
