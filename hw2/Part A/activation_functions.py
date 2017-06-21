from math import exp


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
