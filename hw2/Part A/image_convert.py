from PIL import Image
from os import path, makedirs, remove, listdir
from shutil import rmtree
from scipy.misc import imread
from pickle import dump, HIGHEST_PROTOCOL, load
from scipy.misc import toimage
from numpy import array, uint8

GRAY_IMAGES_FOLDER = 'Gray images'
MAX_PIXEL_VALUE = 255
MIN_PIXEL_VALUE = 0
IMAGE_SHAPE = (256, 256)


class ImageConvert(object):
    def __init__(self, image_source_path, image_size_after_reduce, sub_image_shape, image_mode):
        if not path.isfile(image_source_path):
            print("Image does not exist in '{}' path!".format(image_source_path))
            exit()

        self.__image_source_path = image_source_path
        self.__gray_image_path = path.join(GRAY_IMAGES_FOLDER, path.split(image_source_path)[-1])
        self.__gray_shape = image_size_after_reduce
        self.__image_mode = image_mode
        self.__sub_image_shape = sub_image_shape
        self.__image_shape = IMAGE_SHAPE
        self.__sub_images_data_list = self.__create_sub_images_data()

    def __reduce_size_and_change_image_to_gray(self):
        if not path.isfile(self.__gray_image_path):
            Image.open(self.__image_source_path).convert(self.__image_mode).resize(self.__gray_shape) \
                .save(self.__gray_image_path, optimize=True, quality=95)
            print("Save gray image in '{}' path".format(self.__gray_image_path))

        return imread(self.__gray_image_path).tolist()

    def __create_sub_images_data(self):
        image_data_list = self.__reduce_size_and_change_image_to_gray()
        cols_size = self.__gray_shape[1] // self.__sub_image_shape[1]
        row_start_idx = self.__gray_shape[0] - cols_size * self.__sub_image_shape[0]
        col_start_idx = self.__gray_shape[1] - cols_size * self.__sub_image_shape[1]
        file_idx = 1

        sub_images_data_list = []
        for i in range(row_start_idx, self.__gray_shape[0] - 1, self.__sub_image_shape[0]):
            for j in range(col_start_idx, self.__gray_shape[1] - 1, self.__sub_image_shape[1]):
                sub_images_data_list.append([])
                for row_idx in range(i, i + self.__sub_image_shape[0]):
                    sub_images_data_list[-1].append(image_data_list[row_idx][j: j + self.__sub_image_shape[1]])

                file_idx += 1
        return self.__normalize_sub_image_list(sub_images_data_list, MAX_PIXEL_VALUE, MIN_PIXEL_VALUE)

    @staticmethod
    def __normalize_sub_image_list(sub_images_data_list, max_pixel_value, min_pixel_value):
        max_mix_diff = float(max_pixel_value - min_pixel_value)
        norm_sub_image_data_list = []
        for sub_image_list in sub_images_data_list:
            norm_sub_image_data_list.append([])
            for line in sub_image_list:
                for pixel in line:
                    norm_sub_image_data_list[-1].append((pixel - min_pixel_value) / max_mix_diff)

        return norm_sub_image_data_list

    @staticmethod
    def __abnormal_sub_images_file(sub_images_data_list):
        max_mix_diff = MAX_PIXEL_VALUE - MIN_PIXEL_VALUE

        abnormal_sub_image_data_list = []
        for line in sub_images_data_list:
            abnormal_sub_image_data_list.append([])
            for pixel in line:
                abnormal_sub_image_data_list[-1].append(int(pixel * max_mix_diff + MIN_PIXEL_VALUE))

        return abnormal_sub_image_data_list

    def create_original_image_from_sub_images_data_list(self, sub_images_data_list):
        sub_images_data_list = self.__abnormal_sub_images_file(sub_images_data_list)
        sub_images_list = []
        for sub_image_list in sub_images_data_list:
            sub_images_list.append([])
            for idx, pixel in enumerate(sub_image_list):
                if idx % self.__sub_image_shape[1] == 0:
                    sub_images_list[-1].append([])
                sub_images_list[-1][-1].append(pixel)

        sub_images_data_list = sub_images_list
        sub_image_in_image_rows_ratio = self.__image_shape[0] // self.__sub_image_shape[0]
        image_rows_size = self.__image_shape[0] // self.__sub_image_shape[0] * self.__sub_image_shape[0]
        restored_original_image_list = [[] for _ in range(image_rows_size)]
        row_idx = 0
        cols_size = self.__sub_image_shape[1]
        for part_idx in range(0, len(sub_images_data_list), sub_image_in_image_rows_ratio):
            for j in range(cols_size):
                for i in range(0, len(sub_images_data_list) // sub_image_in_image_rows_ratio):
                    restored_original_image_list[row_idx] = restored_original_image_list[row_idx] \
                                                            + sub_images_data_list[i + part_idx][j]
                row_idx += 1

        return restored_original_image_list

    def get_rotated_image_sub_images_data(self, rotation_angle, image_name):
        restored_original_image_list = self.create_original_image_from_sub_images_data_list(self.get_sub_images_data_list())
        data = array(restored_original_image_list, dtype=uint8)
        img = toimage(data)
        rotated_img = img.rotate(rotation_angle)
        rotated_image_name = image_name + "_" + str(rotation_angle) + "_deg_rotation"
        rotated_image_path = path.join('Images', rotated_image_name + ".png")
        rotated_img.save(rotated_image_path)
        other_converter = ImageConvert(rotated_image_path, self.__image_shape, self.__sub_image_shape, self.__image_mode)
        return other_converter.get_sub_images_data_list()

    @staticmethod
    def show_image_on_screen(restored_original_image_list):
        data = array(restored_original_image_list, dtype=uint8)
        img = toimage(data)
        img.show()

    def get_sub_images_data_list(self):
        return self.__sub_images_data_list

#
# def main():
#     image_convert_obj = ImageConvert('Images\\Lena\\lena.png', (256, 256), (30, 30), 'L')
#     sub_images_data_list = image_convert_obj.get_sub_images_data_list()
#     original_image = image_convert_obj.create_original_image_from_sub_images_data_list(sub_images_data_list)
#     image_convert_obj.show_image_on_screen(original_image)
#     print()
# if __name__ == "__main__":
#     main()
