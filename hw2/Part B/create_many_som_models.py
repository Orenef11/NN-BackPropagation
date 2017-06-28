# coding=utf-8
from create_dataset import KohonenDataBase
from kohonen_network import KohonenNetwork
from time import clock, sleep
from multiprocessing import Pool, cpu_count
from os import path, makedirs
from shutil import rmtree
import logging
from random import uniform
from math import sqrt, cos, sin


X_RANGE_BOUNDS = (0, 1)
Y_RANGE_BOUNDS = (0, 1)
BOTTOM_RANGE = 0
UPPER_RANGE = 1
CENTER = (0, 0)
NUMBER_OF_POINTS = 1000


def __create_networks_structure_list(learning_rates_list, databast_obj_list, network_dimensions, images_folder_path):
    models_structure_tuple = []
    file_count = 1
    for learning_rate in learning_rates_list:
        for database_obj in databast_obj_list:
            models_structure_tuple.append((network_dimensions, database_obj, learning_rate, file_count,
                                           images_folder_path))
            file_count += 1

    return tuple(models_structure_tuple)


def __create_folder_path(folder_path_list):
    for folder_path in folder_path_list:
        if path.isdir(folder_path):
            rmtree(folder_path)
            sleep(0.5)
            makedirs(folder_path)
            print("Create '{}' folder".format(folder_path))


def __calculate_distance(first_point, second_point):
    raw_distance_value = 0
    for first_point_parameter, second_point_parameter in zip(first_point, second_point):
        raw_distance_value += (first_point_parameter - second_point_parameter) ** 2
    return raw_distance_value ** 0.5


def __choose_point_uniformly():
    return tuple((uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1]), uniform(Y_RANGE_BOUNDS[0], Y_RANGE_BOUNDS[1])))


def choose_point_by_x():
    is_x_chosen_successfully = False
    x_value = None
    while is_x_chosen_successfully is False:
        x_value = uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1])
        x_pooling_probability_to_choose = uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1])
        is_x_chosen_successfully = x_value >= x_pooling_probability_to_choose
    y_value = uniform(Y_RANGE_BOUNDS[0], Y_RANGE_BOUNDS[1])
    return tuple((x_value, y_value))


def choose_point_by_distance_to_center():
    is_chosen_point_successfully = False
    point = ()
    while is_chosen_point_successfully is False:
        point = tuple((uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1]), uniform(Y_RANGE_BOUNDS[0], Y_RANGE_BOUNDS[1])))
        distance_from_center = __calculate_distance(point, CENTER)
        probability_to_choose = uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1])
        is_chosen_point_successfully = distance_from_center >= probability_to_choose
    return point


def choose_points_with_radius_of_one_annulus():
    theta = 360 * uniform(BOTTOM_RANGE, UPPER_RANGE)
    dist = sqrt(uniform(BOTTOM_RANGE, UPPER_RANGE) * (0.5**2 - 1**2) + 1**2)
    point = (dist * cos(theta), dist * sin(theta))
    return point


def choose_points_with_radius_of_two_annulus():
    theta = 360 * uniform(BOTTOM_RANGE, UPPER_RANGE)
    dist = sqrt(uniform(BOTTOM_RANGE, UPPER_RANGE) * (1**2 - 2**2) + 2**2)
    point = (dist * cos(theta), dist * sin(theta))
    return point


def create_nn_models_main(images_folder_path):
    start_time = clock()
    # Creates all the possibilities of the information that can be
    logging.info('\nCreates a list of database object according to: ')
    logging.info('\nuniform, distance from x, distance from x, y, and circle.')
    databast_obj_list = [KohonenDataBase(__choose_point_uniformly, NUMBER_OF_POINTS, 'Uniformly Data'),
                         KohonenDataBase(choose_point_by_x, NUMBER_OF_POINTS, 'Data by X'),
                         KohonenDataBase(choose_point_by_distance_to_center, NUMBER_OF_POINTS,
                                         'Data by distance to center'),
                         KohonenDataBase(choose_points_with_radius_of_one_annulus, NUMBER_OF_POINTS,
                                         'Data with radius of onr annulus'),
                         KohonenDataBase(choose_points_with_radius_of_two_annulus, NUMBER_OF_POINTS,
                                         'Data with radius of onr annulus')]
    finish_dataset_time = clock()
    logging.info('\nFinish create sub-images, it took {} seconds'.format(finish_dataset_time - start_time))
    logging.info('\nCreate networks structure list')
    learning_rates_list = [learning_rate / 100.0 for learning_rate in range(5, 31, 5)]
    logging.info('\nCreate networks structure list')
    networks_structure_list = __create_networks_structure_list(learning_rates_list, databast_obj_list, (12, 12),
                                                               images_folder_path)
    logging.info('\nFinish create networks structure list, it took {} seconds'.format(clock() - finish_dataset_time))
    som_networks_training = clock()
    logging.info('\nTraining all networks structure')
    networks_structure_list = networks_structure_list[:2]
    KohonenNetwork(networks_structure_list[0])

    pool_processes = Pool(processes=cpu_count())
    pool_processes.map(KohonenNetwork, networks_structure_list)

    logging.info("\nFinish training, it took {} minutes".format((clock() - som_networks_training) / 60.0))
    logging.info("\nThe running time took {} minutes".format((clock() - start_time) / 60.0))
