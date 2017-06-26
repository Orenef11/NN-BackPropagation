from pickle import load
from os import listdir, path, makedirs
from time import clock, sleep
from shutil import rmtree, copyfile


def best_k_models(models_pickle_folder, best_k_models_folder, best_k=5):
    start_time = clock()
    nn_models_files_list = [path.join(models_pickle_folder, filename) for filename in listdir(models_pickle_folder)]
    k_best_models_tuple_list = []
    for model_path in nn_models_files_list:
        with open(model_path, 'rb') as nn_file:
            nn_model = load(nn_file)
            k_best_models_tuple_list.append((nn_model.error_rate, path.split(model_path)[-1]))
        nn_file.close()
    k_best_models_tuple_list.sort(key=lambda tup: tup[0])

    if path.isdir(best_k_models_folder):
        rmtree(best_k_models_folder)
        sleep(1)
    makedirs(best_k_models_folder)
    for error_and_filename in k_best_models_tuple_list[:best_k]:
        filename = error_and_filename[1]
        copyfile(path.join(models_pickle_folder, filename), path.join(best_k_models_folder, filename))

    print("Total time that taken program to run is {} secs".format(clock() - start_time))

