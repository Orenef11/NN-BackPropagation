def calculate_distance(first_point, second_point):
    raw_distance_value = 0
    for first_point_parameter, second_point_parameter in zip(first_point, second_point):
        raw_distance_value += (first_point_parameter - second_point_parameter) ** 2
    return raw_distance_value ** 0.5
