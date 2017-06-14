from neural_network import Network
from time import clock
from multiprocessing import Pool, cpu_count
from os import path, listdir
from pickle import load
from convert_image import ImageConvert
from activation_functions import *

LENA_IMAGE_PATH = path.join('Images', path.join('Lena', 'lena.png'))
LENA_IMAGE_DES_PATH = path.join(path.join('Sub-images', 'Lena'))
NEURON_OUTPUT_SIZE = 900

IMAGE_MODE = 'L'
IMAGE_SHAPE = (256, 256)
SUB_IMAGE_SHAPE = (30, 30)
DUMP_FILE = 'dump.txt'


def create_networks_structure_list(image_convert_obj):
    models_structure_tuple = []
    file_count = 1
    for learning_rate in [0.1, 0.3, 0.5]:
        for idx, hidden_neuron_size in enumerate(range(50, 901, 50)):
            models_structure_tuple.append([hidden_neuron_size, NEURON_OUTPUT_SIZE, learning_rate, image_convert_obj,
                                          sigmoid_activation_function, sigmoid_activation_function_derivative,
                                           file_count])
            models_structure_tuple.append([hidden_neuron_size, NEURON_OUTPUT_SIZE, learning_rate, image_convert_obj,
                                          bipolar_sigmoid_activation_function,
                                          bipolar_sigmoid_activation_function_derivative, file_count + 1])
            models_structure_tuple.append([hidden_neuron_size, NEURON_OUTPUT_SIZE, learning_rate, image_convert_obj,
                                          hyperbolic_tangent_activation_function,
                                          hyperbolic_tangent_activation_function_derivative, file_count + 2])
            file_count += 3

    return tuple(models_structure_tuple)


def create_networks_structure_list_from_pickle_files(pickle_folder_path):
    networks_structure_list = [[] for _ in range(len(listdir(pickle_folder_path)))]
    for pickle_file in listdir(pickle_folder_path):
        with open(path.join(pickle_folder_path, pickle_file), 'rb') as f:
            networks_structure_list[int(pickle_file.split('-')[1]) - 1] = load(f)

    return networks_structure_list


def main():
    start_time = clock()
    dump_file = open(DUMP_FILE, mode='w', encoding='utf-8')
    dump_file.write("Start time is {}\n".format(start_time))
    # Crate sun-images files
    dump_file.write('Create sub-images split from original image\n')
    image_convert_obj = ImageConvert(LENA_IMAGE_PATH, LENA_IMAGE_DES_PATH, IMAGE_SHAPE, SUB_IMAGE_SHAPE, IMAGE_MODE)
    finish_image_convert_time = clock()
    dump_file.write("Finish create sub-images, it took {} seconds\n".format(finish_image_convert_time - start_time))
    dump_file.write('Create networks structure list\n')
    networks_structure_list = create_networks_structure_list(image_convert_obj)
    dump_file.write('Finish create networks structure list, it took {} seconds\n'
                    .format(clock() - finish_image_convert_time))
    networks_training = clock()
    dump_file.write('Training all networks structure \n')
    pool_processes = Pool(processes=cpu_count() * 25)
    pool_processes.map(Network, networks_structure_list)

    dump_file.write("Finish training, it took {} minutes\n".format((clock() - networks_training) / 60))
    dump_file.write("The running time took {} minutes\n".format((clock() - start_time) / 60))

if __name__ == "__main__":
    main()
