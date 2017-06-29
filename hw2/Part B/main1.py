# coding=utf-8
from create_many_som_models import create_nn_models_main
from best_k_models import best_k_models
from time import sleep, clock
from sys import argv
from os import path, listdir, makedirs, remove
from pickle import load
from image_convert import ImageConvert
from copy import copy
from shutil import rmtree
import matplotlib.pyplot as plt




import logging
from logger import setting_up_logger

MODEL_IMAGES_OUTPUT_FOLDER = 'Images from model'


LENA_IMAGE_PATH = path.join('Images', path.join('Lena', 'lena.png'))
BEST_K_MODELS_FOLDER = 'Best K Models'
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


def create_folders(folder_path_list):
    """
    Creates the folders from variable 'folder_path_list', if existing, deletes and recreates.
    :param folder_path_list: List of folder paths to create them

    """
    for folder_path in folder_path_list:
        if path.isdir(folder_path):
            rmtree(folder_path)
            sleep(0.5)
        makedirs(folder_path)
        logging.info("\nCreate '{}' folder".format(folder_path))


def main():
    """
    Create some models for the same configuration of the network.
    Using multi-processors to run parallel.
    Save for each model the images of network growth along with the dump file.
    """

    start_time = clock()
    # if path.isfile('som_dump.log'):
    #     remove('som_dump.log')
    #     sleep(0.5)
    setting_up_logger('debug', 'info', 'som_dump.log')
    logging.info("\nCreate new folders")
    create_models_flag = True
    # create_folders([MODEL_IMAGES_OUTPUT_FOLDER])
    if create_models_flag:
        create_nn_models_main(MODEL_IMAGES_OUTPUT_FOLDER)
    logging.info("\nThe running time of the program is {} minutes".format((clock() - start_time) / 60.0))


if __name__ == '__main__':
    main()

