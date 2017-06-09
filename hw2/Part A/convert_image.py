from PIL import Image
from os import listdir, path, makedirs
from time import sleep, time
from shutil import rmtree
from scipy.misc import imread

IMAGES_FOLDER = 'Images'
GRAY_IMAGES_FOLDER = 'Gray images'
SUB_IMAGES_FOLDER = 'Sub-images'
IMAGE_COLOR = 'L'
IMAGE_SHAPE = (256, 256)
SUB_IMAGE_SHAPE = (30, 30)
IMAGE_FORMAT = 'png'


class ImageConvert(object):
    def __init__(self, image_source_folder, image_des_folder, image_size_after_reduce, image_format):
        if not path.isdir(image_source_folder):
            makedirs(image_source_folder)
            print("There are no images in the '{}' folder!".format(image_source_folder))
            exit()
        if not path.isdir(image_des_folder):
            makedirs(image_des_folder)
            print("Create '{}' folder".format(image_des_folder))

        self.__source_images_path_list = [path.join(image_source_folder, image_name) for image_name in
                                          listdir(image_source_folder)]
        self.__des_images_path_list = [path.join(image_des_folder, image_name.split('.')[0]) + '.' + image_format
                                       for image_name in listdir(image_source_folder)]
        self.__image_shape = image_size_after_reduce

    def reduce_and_change_color_from_source_folder(self, image_color):
        for src_image_path, des_image_path in zip(self.__source_images_path_list, self.__des_images_path_list):
            image_obj = Image.open(src_image_path).convert(image_color)
            image_obj = image_obj.resize(IMAGE_SHAPE, Image.ANTIALIAS)
            image_obj.save(des_image_path, optimize=True, quality=95)

    def create_sub_ascii_image(self, folder_path, sub_image_shape):
        if not path.isdir(folder_path):
            makedirs(folder_path)
            print("Create '{}' folder".format(folder_path))
        for image_idx, des_image_path in enumerate(self.__des_images_path_list):
            with Image.open(des_image_path) as img:
                source_image_shape = img.size

            # # Testing
            # source_image_shape = IMAGE_SHAPE
            # image_lines_list = []
            # rows_size = source_image_shape[0]
            # for i in range(0, rows_size):
            #     image_lines_list.append([])
            #     for j in range(1, IMAGE_SHAPE[1] + 1):
            #         image_lines_list[-1].append(i * rows_size + j)

            cols_size = int(source_image_shape[1] / sub_image_shape[1])
            row_start_idx = source_image_shape[0] - cols_size * sub_image_shape[0]
            col_start_idx = source_image_shape[1] - cols_size * sub_image_shape[1]
            image_lines_list = imread(des_image_path).tolist()
            file_idx = 1

            image_folder_path = path.join(folder_path, path.split(des_image_path)[1].split('.')[0])
            if path.isdir(image_folder_path):
                rmtree(image_folder_path)
                # sleep(1)
            makedirs(image_folder_path)
            for i in range(row_start_idx, source_image_shape[0] - 1, sub_image_shape[0]):
                for j in range(col_start_idx, source_image_shape[1] - 1, sub_image_shape[1]):
                    file_path = path.join(image_folder_path, str(file_idx) + '.txt')
                    with open(file_path, 'w') as f:
                        for row_idx in range(i, i + sub_image_shape[0]):
                            line = list(map(str, image_lines_list[row_idx][j: j + sub_image_shape[1]]))
                            f.write('\t'.join(line) + '\n')
                    print("Create '{}' file".format(file_path))
                    file_idx += 1


def read_sub_images_file(image_folder_path):
    image_list = []
    for image_name in listdir(image_folder_path):
        with open(path.join(image_folder_path, image_name), 'r') as f:
            sub_image_list = [line.replace('\n', '') for line in f.readlines()]
        image_list.append([list(map(int, line.split('\t'))) for line in sub_image_list])
    return image_list


def main():
    for folder_name in listdir(IMAGES_FOLDER):
        images_convert_obj = ImageConvert(path.join(IMAGES_FOLDER, folder_name),
                                          path.join(GRAY_IMAGES_FOLDER, folder_name), IMAGE_SHAPE, IMAGE_FORMAT)
        images_convert_obj.reduce_and_change_color_from_source_folder(IMAGE_COLOR)
        images_convert_obj.create_sub_ascii_image(path.join(SUB_IMAGES_FOLDER, folder_name), SUB_IMAGE_SHAPE)


    # Contains in each row the sub-picture derived from the big picture.
    image_path = path.join(path.join(SUB_IMAGES_FOLDER, 'Lena'), 'Lena')
    image_list = read_sub_images_file(image_path)
    print(image_list)

if __name__ == "__main__":
    main()
