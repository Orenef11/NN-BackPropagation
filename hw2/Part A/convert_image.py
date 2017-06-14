from PIL import Image
from os import path, makedirs, remove, listdir
from shutil import rmtree
from scipy.misc import imread
from pickle import dump, HIGHEST_PROTOCOL, load
from scipy.misc import toimage
from numpy import array, uint8

GRAY_IMAGES_FOLDER = 'Gray images'
IMAGE_SHAPE = (256, 256)


class ImageConvert(object):
    def __init__(self, image_source_path, image_des_foldr_path, image_size_after_reduce, sub_image_shape, image_mode):
        if not path.isfile(image_source_path):
            print("Image does not exist in '{}' path!".format(image_source_path))
            exit()

        file_name = path.split(image_source_path)[1]
        folder_name = path.split(path.split(image_source_path)[0])[1]
        self.__image_source_path = image_source_path
        self.__gray_image_path = path.join(GRAY_IMAGES_FOLDER, path.join(folder_name, file_name))
        self.__image_regular_des_path = path.join(image_des_foldr_path, path.join("regular", file_name.split('.')[0]))
        self.__image_ascii_des_path = path.join(image_des_foldr_path, path.join("ascii", file_name.split('.')[0]))

        self.__image_obj = None
        self.__gray_shape = image_size_after_reduce
        self.__sub_image_shape = sub_image_shape
        self.__image_shape = IMAGE_SHAPE
        self.__image_mode = image_mode
        self.__max_pixel_value = 0
        self.__min_pixel_value = 0

        self.__initialize_and_crate_images_folder()

    def __initialize_and_crate_images_folder(self):
        # Creates the folder that contains all regular sub-images of the original image
        if path.isdir(self.__image_regular_des_path):
            rmtree(self.__image_regular_des_path)
        makedirs(self.__image_regular_des_path)
        print("Create '{}' folder".format(self.__image_regular_des_path))

        # Creates the folder that contains all ascii sub-images of the original image
        if path.isdir(self.__image_ascii_des_path):
            rmtree(self.__image_ascii_des_path)
        makedirs(self.__image_ascii_des_path)
        print("Create '{}' folder".format(self.__image_ascii_des_path))

        # Creates the folder that contains all ascii sub-images of the original image
        gray_folder_path = path.split(self.__gray_image_path)[0]
        if not path.isdir(gray_folder_path):
            makedirs(gray_folder_path)
            print("Create '{}' folder".format(gray_folder_path))
        if path.isfile(self.__gray_image_path):
            remove(self.__gray_image_path)

        # Creates a gray image and reduces its size according to the new size
        self.__reduce_and_change_color_from_source_folder()
        self.__image_obj = imread(self.__gray_image_path)
        self.__max_pixel_value, self.__min_pixel_value = self.__image_obj.max(), self.__image_obj.min()

        # Create regular & ascii sub-images files
        self.__create_sub_image(self.__image_regular_des_path)
        self.__create_sub_image(self.__image_ascii_des_path, True)
        del self.__image_obj

    def __reduce_and_change_color_from_source_folder(self):
        if path.isfile(self.__gray_image_path):
            remove(self.__gray_image_path)
        Image.open(self.__image_source_path).convert(self.__image_mode).resize(self.__gray_shape)\
            .save(self.__gray_image_path, optimize=True, quality=95)
        print("Save gray image in '{}' path".format(self.__gray_image_path))

    def __create_sub_image(self, sub_image_folder_path, ascii_mode=False):
        cols_size = self.__gray_shape[1] // self.__sub_image_shape[1]
        row_start_idx = self.__gray_shape[0] - cols_size * self.__sub_image_shape[0]
        col_start_idx = self.__gray_shape[1] - cols_size * self.__sub_image_shape[1]
        image_lines_list = self.__image_obj.tolist()
        if ascii_mode is True:
            image_lines_list = [list(map(chr, line)) for line in image_lines_list]
        file_idx = 1

        for i in range(row_start_idx, self.__gray_shape[0] - 1, self.__sub_image_shape[0]):
            for j in range(col_start_idx, self.__gray_shape[1] - 1, self.__sub_image_shape[1]):
                file_path = path.join(sub_image_folder_path, str(file_idx) + '.txt')
                sub_image_list = []
                for row_idx in range(i, i + self.__sub_image_shape[0]):
                    sub_image_list.append(list(map(str, image_lines_list[row_idx][j: j + self.__sub_image_shape[1]])))

                with open(file_path, 'wb') as f:
                    dump(sub_image_list, f, HIGHEST_PROTOCOL)
                file_idx += 1

        print("Finished dividing the '{}' image into sub-images".format(sub_image_folder_path))

    def read_sub_images_file(self, image_folder_path):
        image_data_list = []
        for idx in range(1, len(listdir(image_folder_path)) + 1):
            with open(path.join(image_folder_path, str(idx) + '.txt'), 'rb') as f:
                image_data_list.append(load(f))
        image_data_list = [self.__normalize_sub_image_list(sub_image_list) for sub_image_list in image_data_list]
        return image_data_list

    def __abnormal_sub_images_file(self, sub_images_data_list):
        max_mix_diff = self.__max_pixel_value - self.__min_pixel_value

        abnormal_sub_image_data_list = []
        for line in sub_images_data_list:
            abnormal_sub_image_data_list.append([])
            for pixel in line:
                abnormal_sub_image_data_list[-1].append(int(pixel * max_mix_diff + self.__min_pixel_value))

        return abnormal_sub_image_data_list

    def __normalize_sub_image_list(self, sub_images_data_list):
        max_mix_diff = self.__max_pixel_value - self.__min_pixel_value
        sub_images_data_list = [list(map(int, line)) for line in sub_images_data_list]
        norm_sub_image_data_list = []
        for line in sub_images_data_list:
            for pixel in line:
                norm_sub_image_data_list.append((pixel - self.__min_pixel_value) / max_mix_diff)

        return norm_sub_image_data_list

    def create_original_image_from_sub_images_file(self, sub_images_data_list):
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

    @staticmethod
    def show_restored_image_sub_images(restored_original_image_list):
        data = array(restored_original_image_list, dtype=uint8)
        img = toimage(data)
        img.show()

    @property
    def image_regular_des_path(self):
        return self.__image_regular_des_path

    @property
    def image_ascii_des_path(self):
        return self.__image_ascii_des_path


# def main():
#     IMAGES_FOLDER = 'Images'
#     SUB_IMAGES_FOLDER = 'Sub-images'
#     IMAGE_MODE = 'L'
#     SUB_IMAGE_SHAPE = (30, 30)
#     IMAGE_FORMAT = 'png'
#     SUB_IMAGE_FILE_INFO = 'info.txt'
#     for folder_name in listdir(IMAGES_FOLDER):
#         image_des_foldr_path = path.join(SUB_IMAGES_FOLDER, folder_name)
#         for image_name in listdir(path.join(IMAGES_FOLDER, folder_name)):
#             image_source_path = path.join(path.join(IMAGES_FOLDER, folder_name), image_name)
#             image_convert_obj = ImageConvert(image_source_path, image_des_foldr_path, IMAGE_SHAPE, SUB_IMAGE_SHAPE,
#                                              IMAGE_MODE)
#             image_data_list = image_convert_obj.read_sub_images_file(image_convert_obj.image_regular_des_path)
#             original_image = image_convert_obj.create_original_image_from_sub_images_file(image_data_list)
#             image_convert_obj.show_restored_image_sub_images(original_image)
#
#
# if __name__ == "__main__":
#     main()
