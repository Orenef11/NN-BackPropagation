from neural_network import Network
from time import clock, sleep
from multiprocessing import Pool, cpu_count
from os import path, makedirs
from image_convert import ImageConvert
from activation_functions import *
from shutil import rmtree


def __create_networks_structure_list(image_convert_obj, neurons_output_size, learning_rate_list, hidden_size_list):
    models_structure_tuple = []
    file_count = 1
    for learning_rate in learning_rate_list:
        for hidden_neuron_size in hidden_size_list:
            models_structure_tuple.append([hidden_neuron_size, neurons_output_size, learning_rate, image_convert_obj,
                                          sigmoid_activation_function, sigmoid_activation_function_derivative,
                                           file_count])
            models_structure_tuple.append([hidden_neuron_size, neurons_output_size, learning_rate, image_convert_obj,
                                          bipolar_sigmoid_activation_function,
                                          bipolar_sigmoid_activation_function_derivative, file_count + 1])
            models_structure_tuple.append([hidden_neuron_size, neurons_output_size, learning_rate, image_convert_obj,
                                          hyperbolic_tangent_activation_function,
                                          hyperbolic_tangent_activation_function_derivative, file_count + 2])
            file_count += 3

    return tuple(models_structure_tuple)


def __create_folder_path(folder_path_list):
    for folder_path in folder_path_list:
        if path.isdir(folder_path):
            rmtree(folder_path)
            sleep(0.5)
            makedirs(folder_path)
            print("Create '{}' folder".format(folder_path))


def create_nn_models_main(lena_image_path, image_shape_tuple, sub_image_shape_tuple, image_mode,
                          neurons_output_size, learning_rate_list, hidden_size_list):
    start_time = clock()
    # Crate sun-images files
    print('Create sub-images split from original image')
    image_convert_obj = ImageConvert(lena_image_path, image_shape_tuple, sub_image_shape_tuple, image_mode)
    finish_image_convert_time = clock()
    print("Finish create sub-images, it took {} seconds".format(finish_image_convert_time - start_time))
    print('Create networks structure list')
    networks_structure_list = __create_networks_structure_list(image_convert_obj, neurons_output_size,
                                                               learning_rate_list, hidden_size_list, )
    print('Finish create networks structure list, it took {} seconds'
          .format(clock() - finish_image_convert_time))
    networks_training = clock()
    print('Training all networks structure ')
    pool_processes = Pool(processes=cpu_count())
    pool_processes.map(Network, networks_structure_list)

    print("Finish training, it took {} minutes".format((clock() - networks_training) / 60))
    print("The running time took {} minutes".format((clock() - start_time) / 60))
