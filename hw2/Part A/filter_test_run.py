def calculate_distance(sub_picture_values, network_outputs):
    values_sum = 0
    for sub_picture_value, network_output in zip(sub_picture_values, network_outputs):
        values_sum += (sub_picture_value - network_output) ** 2
    return values_sum ** 0.5


def filter_relative_distances(sub_picture_values, network_outputs, epsilon):
    network_output_size = len(network_outputs)
    successes = 0
    for sub_picture_value, network_output in zip(sub_picture_values, network_outputs):
        if abs(sub_picture_value - network_output) <= epsilon:
            successes += 1
    return successes / float(network_output_size)

with open("trained_networks\\model_num_-292-_hidden_size_-99-_learning_rate_-0.1.pickle", 'rb') as model_file:
    network = load(model_file)

larry_picture_converter = ImageConvert(path.join('Images', path.join('Person', 'larry.png')), (256, 256), (30, 30), 'L')
larry_sub_pictures = larry_picture.get_sub_images_data_list()

person_testing_results = []
for larry_sub_picture in larry_sub_pictures:
    network.update_input_layer_output_values(larry_sub_picture)
    person_testing_results.append(network._Network__calculate_net_output())

distance_result = 0
filter_result = []
for person_testing_result, larry_sub_picture in zip(person_testing_results, larry_sub_pictures):
    distance_result += calculate_distance(person_testing_result, larry_sub_picture)
    filter_result.append(filter_relative_distances(person_testing_result, larry_sub_picture))
print("CALCULATE DISTANCE RESULT = ", distance_result)
print("FILTER RELATIVE DISTANCES RESULT = ", sum(filter_result) / 64.0)    

print("displaying original larry picture")
larry_restored_original_image_list = larry_picture_converter.create_original_image_from_sub_images_data_list(person_testing_results)
larry_picture_converter.show_image_on_screen(larry_restored_original_image_list)

print("displaying larry picture after network test run")
larry_actual_original = larry_picture_converter.create_original_image_from_sub_images_data_list(larry_sub_pictures)
larry_picture_converter.show_image_on_screen(larry_actual_original)
