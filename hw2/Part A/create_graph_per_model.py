import matplotlib.pyplot as plt
from os import path, makedirs
MODELS_DUMPS_FILES = 'NN - models dump files'
MODELS_PLOT = 'Models plot'


def main():
    if not path.isdir(MODELS_PLOT):
        makedirs(MODELS_PLOT)
    file_path = path.join(MODELS_DUMPS_FILES, '262.txt')
    epoch_values_list = []
    error_rate_list = []
    with open(file_path) as f:
        for line in f.readlines():
            if ',' in line and len(line.split(',')) == 3:
                epoch_values_list.append(line.split(',')[0].split(' ')[-1])
                error_rate_list.append(line.split(',')[1].split(' ')[-1])
    epoch_values_list = list(map(int, epoch_values_list))
    error_rate_list = list(map(float, error_rate_list))
    plt.plot(epoch_values_list, error_rate_list)
    plt.xlabel('epoch number')
    plt.ylabel('error rate')
    plt.axis([0, len(epoch_values_list), 0, int(max(error_rate_list))])
    model_name = path.split(file_path)[-1].split('.')[0]
    plt.title('Model ' + model_name)
    plt.savefig(path.join(MODELS_PLOT, model_name))
    plt.clf()


if __name__ == '__main__':
    main()
