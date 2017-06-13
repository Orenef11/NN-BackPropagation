def normailze_sample_data(data_row):
        normalized_data_row = []
        min_value = min(data_row)
        max_value = max(data_row)
        for field_index in range(len(data_row)):
            if min_value != max_value:
                data_row_values_max_difference = float(max_value - min_value)
                normalized_data_row.append((data_row[field_index] - min_value) / data_row_values_max_difference)
            else:
                data_norm = (sum(map(lambda x: x ** 2, data_row))) ** 0.5
                normalized_data_row.append(data_row[field_index] / data_norm)
        return normalized_data_row


def normalize_image_data(image, min_pixel_value, max_pixel_value):
    diff_min_and_max = max_pixel_value - min_pixel_value
    image = [(pixel - min_pixel_value) / diff_min_and_max for pixel in [row for row in image]]
    return image
