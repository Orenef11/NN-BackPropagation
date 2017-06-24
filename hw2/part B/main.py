from random import uniform
from dataset.create_data import KohonenDataBase

X_RANGE_BOUNDS = (0, 1)
Y_RANGE_BOUNDS = (0, 1)
CENTER = (0, 0)
NUMBER_OF_POINTS = 4000


def _calculate_distance(first_point, second_point):
    raw_distance_value = 0
    for first_point_parameter, second_point_parameter in zip(first_point, second_point):
        raw_distance_value += (first_point_parameter - second_point_parameter) ** 2
    return raw_distance_value ** 0.5


def choose_point_uniformly():
    return tuple((uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1]), uniform(Y_RANGE_BOUNDS[0], Y_RANGE_BOUNDS[1])))


def choose_point_by_x():
    is_x_chosen_successfully = False
    while is_x_chosen_successfully is False:
        x_value = uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1])
        x_pooling_probability_to_choose = uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1])
        is_x_chosen_successfully = x_value >= x_pooling_probability_to_choose
    y_value = uniform(Y_RANGE_BOUNDS[0], Y_RANGE_BOUNDS[1])
    return tuple((x_value, y_value))


def choose_point_by_distance_to_center():
    is_chosen_point_successfully = False
    while is_chosen_point_successfully is False:
        point = tuple((uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1]), uniform(Y_RANGE_BOUNDS[0], Y_RANGE_BOUNDS[1])))
        distance_from_center = _calculate_distance(point, CENTER)
        probability_to_choose = uniform(X_RANGE_BOUNDS[0], X_RANGE_BOUNDS[1])
        is_chosen_point_successfully = distance_from_center >= probability_to_choose
    return point


def main():
    database = KohonenDataBase(choose_point_uniformly, NUMBER_OF_POINTS)
    database.generate_data_points()
    database.draw_data()

if __name__ == '__main__':
    main()
