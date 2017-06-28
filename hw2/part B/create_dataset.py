# coding=utf-8
import matplotlib.pylab as plt
import logging

class KohonenDataBase(object):
    def __init__(self, kernel_function, number_of_points, dataset_type):
        self.__data_points = []
        self.__kernel_function = kernel_function
        self.__number_of_points = number_of_points
        self.__generate_data_points()
        self.__axis_range = self.__get_axis_plot_range()
        self.__dataset_type = dataset_type

    def __generate_data_points(self):
        logging.debug("\nCreate the database")
        for _ in range(self.__number_of_points):
            # create point object
            self.__data_points.append(self.__kernel_function())
        logging.debug("\n<DATABASE LOGGER> finished the database points generation")

    def __get_axis_plot_range(self):
        min_max_list = tuple(map(sorted, zip(*self.__data_points)))
        x_min, x_max, y_min, y_max = min_max_list[0][0], min_max_list[0][-1], min_max_list[1][0], min_max_list[1][-1]
        x_max, y_max = x_max * 1.1, y_max * 1.1
        if x_min < 1:
            x_min -= x_max * 0.1
        else:
            x_min += x_max * 0.1
        if y_min < 1:
            y_min -= y_max * 0.1
        else:
            y_min += y_max * 0.1
        return [x_min, x_max, y_min, y_max]

    def get_data_points(self):
        return self.__data_points

    @property
    def axis_range(self):
        return self.__axis_range

    @property
    def dataset_type(self):
        return self.__dataset_type

    def draw_data(self, save_image_flag=False, image_path=None):
        plt.title(self.__dataset_type)
        plt.axis(self.__axis_range)
        for point in self.__data_points:
            plt.plot(point[0], point[1], "ro")
        if not save_image_flag:
            plt.show()
        elif image_path is not None:
            plt.savefig(image_path)

        plt.clf()
