from matplotlib.pylab import plot, axis, show


class KohonenDataBase(object):
    def __init__(self, kernel_function, number_of_points):
        self._data_points = []
        self._kernel_function = kernel_function
        self._number_of_points = number_of_points

    def generate_data_points(self):
        for _ in range(self._number_of_points):
            # create point object
            self._data_points.append(self._kernel_function())
        print "<DATABASE LOGGER> finished the database points generation"

    def get_data_points(self):
        return self._data_points

    def draw_data(self):
        axis([-3, 3, -3, 3])
        for point in self._data_points:
            plot(point[0], point[1], "ro")
        show()
