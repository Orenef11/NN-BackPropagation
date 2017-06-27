from create_many_nn_models import create_nn_models_main
from best_k_models import best_k_models
from time import sleep
from sys import argv
from os import path, listdir, makedirs
from pickle import load
from image_convert import ImageConvert
from copy import copy
from shutil import rmtree
import matplotlib.pyplot as plt


LENA_IMAGE_PATH = path.join('Images', path.join('Lena', 'lena.png'))
BEST_K_MODELS_FOLDER = 'Best K Models'
MODELS_PICKLE_FOLDER = 'Models structure'
MODEL_IMAGES_OUTPUT_FOLDER = 'Images from model'
MODELS_DUMPS_FILES = 'NN - models dump files'
MODELS_PLOT = 'Models plot'
IMAGES_FOLDER = 'Images'
SLEEP_TIME = 0.5
NEURON_OUTPUT_SIZE = 900
IMAGE_MODE = 'L'
DUMP_FORMAT = '.txt'
IMAGE_SHAPE = (256, 256)
SUB_IMAGE_SHAPE = (30, 30)
APPROXIMATION_EPSILON = 0.15
PARALLEL_MODELS_SIZE = 3


def __filter_relative_distances(sub_picture_values, network_outputs, epsilon):
    network_output_size = len(network_outputs)
    successes = 0
    for sub_picture_value, network_output in zip(sub_picture_values, network_outputs):
        if abs(sub_picture_value - network_output) <= epsilon:
            successes += 1
    return successes / float(network_output_size)


def analysis_of_models(best_k_models_folder_path, save_images_from_models_folder_path, images_source_folder_path,
                       models_dumps_files_folder):
    activation_functions_names_list = ['sigmoid', 'bipolar', 'hyperbolic']

    for model_file_name in listdir(best_k_models_folder_path):
        with open(path.join(best_k_models_folder_path, model_file_name), 'rb') as model_file:
            model_folder_path = path.join(save_images_from_models_folder_path, model_file_name.split('.')[0])
            model_name = str(model_file_name.split('-')[1]) + DUMP_FORMAT
            parallel_model_names_list = \
                __get_parallel_model_names_with_the_same_parameter(model_name, models_dumps_files_folder)
            __create_plot_to_models(parallel_model_names_list, activation_functions_names_list,
                                    model_name.split('.')[0])
            makedirs(model_folder_path)
            # Create Network obj
            model = load(model_file)
            dump_file = open(path.join(model_folder_path, 'dump.txt'), mode='w', encoding='utf-8')
            for image_obj_folder in listdir(images_source_folder_path):
                # The full path of the image folder
                folder_path = path.join(images_source_folder_path, image_obj_folder)
                for image_file_name in listdir(folder_path):
                    # The full path of the image
                    image_path = path.join(folder_path, image_file_name)
                    image_obj = ImageConvert(image_path, IMAGE_SHAPE, SUB_IMAGE_SHAPE, IMAGE_MODE)
                    save_image_path = path.join(model_folder_path, 'original - ' + image_file_name)
                    image_obj.show_image_on_screen(image_obj.get_sub_images_data_list(), save_image_path, False)
                    print("Save original image in '{}'".format(save_image_path))
                    results_from_model_list = []
                    model_copy = copy(model)
                    distance_result = 0.0
                    for sub_image in image_obj.get_sub_images_data_list():
                        model_copy.update_input_layer_neurons_value(sub_image)
                        model_output = model_copy.calculate_net_output()
                        distance_result += __filter_relative_distances(sub_image, model_output, APPROXIMATION_EPSILON)
                        results_from_model_list.append(model_output)
                    dump_file.write(image_file_name + '\n')
                    dump_file.write("Model predict {}\n".format(distance_result /
                                                                float(len(image_obj.get_sub_images_data_list()))))
                    save_image_path = path.join(model_folder_path, 'model predict - ' + image_file_name)
                    image_obj.show_image_on_screen(results_from_model_list, save_image_path, False)
                    print("Save predict image in '{}'".format(save_image_path))
            dump_file.close()


def __create_plot_to_models(models_names_list, activation_functions_names_list, plot_name):
    x_max_size, y_max_size = 0, 0
    for idx, file_path in enumerate(models_names_list):
        epoch_values_list = []
        error_rate_list = []
        with open(file_path) as f:
            for line in f.readlines():
                if ',' in line and len(line.split(',')) == 3:
                    epoch_values_list.append(line.split(',')[0].split(' ')[-1])
                    error_rate_list.append(line.split(',')[1].split(' ')[-1])
        epoch_values_list = list(map(int, epoch_values_list))
        error_rate_list = list(map(float, error_rate_list))
        plt.plot(epoch_values_list, error_rate_list, label=activation_functions_names_list[idx])

        if x_max_size < max(epoch_values_list):
            x_max_size = max(epoch_values_list)
        if y_max_size < max(error_rate_list):
            y_max_size = max(error_rate_list)
    plt.axis([0, x_max_size, 0, y_max_size])
    plt.xlabel('epoch number')
    plt.ylabel('error rate')
    plt.title('Model ' + plot_name)
    plt.legend()
    plt.savefig(path.join(MODELS_PLOT, plot_name))
    print("Save plot in '{}'".format(path.join(MODELS_PLOT, plot_name)))
    plt.clf()


def __get_parallel_model_names_with_the_same_parameter(model_name, models_dumps_folder):
    parallel_models_list = []
    model_number = int(model_name.split('.')[0]) - 1
    model_number -= int(model_number) % PARALLEL_MODELS_SIZE
    for model_name in range(model_number, model_number + PARALLEL_MODELS_SIZE, 1):
        parallel_models_list.append(model_name)

    parallel_models_path_list = []
    for model_name in parallel_models_list:
        model_path = path.join(models_dumps_folder, str(model_name + 1) + DUMP_FORMAT)
        if path.isfile(model_path):
            parallel_models_path_list.append(model_path)
        else:
            parallel_models_list.append(None)

    return parallel_models_path_list


def main():
    if len(argv) < 4:
        print("The number of arguments entered is incorrect!")
        print("Usage: Python hw1.py <create_models_flag> <choosing_best_modules_flag> <best_k_models_size>")
    create_models_flag = argv[1].lower() == 'true'
    choosing_best_modules_flag = argv[2].lower() == 'true'
    best_k = int(argv[3])

    if not path.isdir(IMAGES_FOLDER):
        print("FolderNotExistError: The '{}' does not exist".format(IMAGES_FOLDER))
    if not path.isdir(BEST_K_MODELS_FOLDER):
        print("FolderNotExistError: The '{}' does not exist".format(BEST_K_MODELS_FOLDER))

    if create_models_flag:
        neurons_output_size = 900
        learning_rate_list = [i / 10 for i in range(1, 11)]
        hidden_size_list = [i for i in range(1, 101)]
        create_nn_models_main(LENA_IMAGE_PATH, IMAGE_SHAPE, SUB_IMAGE_SHAPE, IMAGE_MODE, neurons_output_size,
                              learning_rate_list, hidden_size_list)
    if choosing_best_modules_flag:
        best_k_models(MODELS_PICKLE_FOLDER, BEST_K_MODELS_FOLDER, best_k)

    if path.isdir(MODELS_PLOT):
        rmtree(MODELS_PLOT)
        sleep(SLEEP_TIME)
    makedirs(MODELS_PLOT)
    if path.isdir(MODEL_IMAGES_OUTPUT_FOLDER):
        rmtree(MODEL_IMAGES_OUTPUT_FOLDER)
        sleep(SLEEP_TIME)
    makedirs(MODEL_IMAGES_OUTPUT_FOLDER)

    analysis_of_models(BEST_K_MODELS_FOLDER, MODEL_IMAGES_OUTPUT_FOLDER, IMAGES_FOLDER, MODELS_DUMPS_FILES)

if __name__ == '__main__':
    main()

