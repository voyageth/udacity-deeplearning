import gc
import os
import os.path
import time
import random

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
from skimage.transform import resize
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter


class MultiDigitSVHNData(object):
    def __init__(self, digit_count=5, input_total_data_count=1, 
                image_size=32, use_standard_score=False, add_margin=False, margin_ratio=0.1,
                random_position_count=0, random_rotate_count=0, apply_gaussian_filter=False
                ):
        random.seed(42)
        self.use_standard_score = use_standard_score
        self.add_margin = add_margin
        self.margin_ratio = margin_ratio
        self.random_position_count = random_position_count
        self.random_rotate_count = random_rotate_count
        self.apply_gaussian_filter = apply_gaussian_filter
        self.digit_count = digit_count
        self.image_width = image_size
        self.image_height = image_size
        self.target_digit_num_labels = 10
        self.num_channels = 3

        self.input_total_data_count = input_total_data_count
        self.total_data_count = input_total_data_count  # ds.num_examples
        self.update_data_counts()

        self.train_data = None
        self.train_label_length = None
        self.train_label_digits = None
        self.validation_data = None
        self.validation_label_length = None
        self.validation_label_digits = None
        self.test_data = None
        self.test_label_length = None
        self.test_label_digits = None

    def update_data_counts(self):
        self.train_data_count = int(0.9 * self.total_data_count)
        self.validation_data_count = int(0.05 * self.total_data_count)
        self.test_data_count = self.total_data_count - self.train_data_count - self.validation_data_count

    def print_image(self, image):
        sample_image = image
        plt.figure()
        plt.imshow(sample_image)  # display it
        gc.collect()

    @staticmethod
    def maybe_pickle(digit_count, total_data_count, image_size=54, force=False, 
                     use_standard_score=False, add_margin=False, margin_ratio=0.1,
                     random_position_count=0, random_rotate_count=0, 
                     apply_gaussian_filter=False):
        filename = 'svhn/reader/svhn_' + str(digit_count) \
            + '_' + str(image_size) \
            + '_' + str(total_data_count) \
            + '_' + str(use_standard_score) \
            + '_' + str(add_margin) \
            + '_' + str(margin_ratio) \
            + '_' + str(apply_gaussian_filter) \
        
        if random_position_count > 0:
            filename += '_position' + str(random_position_count) \
        
        if random_rotate_count > 0:
            filename += '_rotate' + str(random_rotate_count) \
            
        filename += '.pickle'

        if os.path.exists(filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % filename)
            svhn_data = MultiDigitSVHNData.load_pickle(filename)
        else:
            print('Pickling %s.' % filename)
            svhn_data = MultiDigitSVHNData(digit_count=digit_count, input_total_data_count=total_data_count,
                                            image_size=image_size, use_standard_score=use_standard_score, add_margin=add_margin,
                                            random_position_count=random_position_count, random_rotate_count=random_rotate_count,
                                            margin_ratio=margin_ratio, apply_gaussian_filter=apply_gaussian_filter)
            svhn_data.load_data()
            try:
                MultiDigitSVHNData.write_pickle(filename, svhn_data)
            except Exception as e:
                print('Unable to save data to', filename, ':', e)
        return svhn_data

    @staticmethod
    def write_pickle(filename, data):
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def reformat_target_length(self, target_length):
        return (np.arange(self.digit_count + 1) == target_length).astype(np.float32)

    def reformat_target_digits(self, target_digits):
        return (np.arange(10) == target_digits).astype(np.float32)
    
    def generate_data(self, rows, total_data_index, target_image_length, data_index, target_image,top,bottom,left,right,rotate_rand=False):
        self.total_label[total_data_index][0] = target_image_length

        for digit_index in range(0, target_image_length):
            self.total_label[total_data_index][1 + digit_index] = rows[1][data_index][digit_index][0]
            
        rand_rotate_angle = 0
        if rotate_rand:
            if random.choice([True, False]):
                rand_rotate_angle = random.randrange(1,91)
            else:
                rand_rotate_angle = random.randrange(-91, -1)

        for channel_index in range(0, self.num_channels):
            if self.use_standard_score:
                mean = np.mean(target_image[:][:][channel_index], dtype='float32')
                std = np.std(target_image[:][:][channel_index], dtype='float32', ddof=1)
                if std < 1e-4:std = 1
                target_image[:][:][channel_index] = ((target_image[:][:][channel_index] - mean) / std).astype('uint8')
            
            croped = target_image[channel_index][top:bottom, left:right]
            if rotate_rand:
                croped = rotate(croped, rand_rotate_angle)
            resized_image = resize(croped,output_shape=[self.image_width, self.image_height])
            for image_height_index in range(0, self.image_height):
                for image_width_index in range(0, self.image_width):
                    self.total_data[total_data_index][image_height_index][image_width_index][channel_index] = \
                        resized_image[image_height_index][image_width_index] * 255

        if self.apply_gaussian_filter:
            self.total_data[total_data_index] = gaussian_filter(self.total_data[total_data_index], 1)
        total_data_index += 1
        
        return total_data_index

    def load_data(self):
        from fuel.config_parser import config
        from fuel.datasets.svhn import SVHN
        config.data_path = './svhn/converted'
        ds = SVHN(which_format=1, which_sets=('train', 'test', 'extra'))
        fetched_total_data_count = min(ds.num_examples, self.input_total_data_count)
        rows = np.array(ds.get_data(state=ds.open(), request=range(0, fetched_total_data_count)))
        rows_len = len(rows)
        print("rows[5][0]", rows[5][0].shape, rows[5][0].dtype)

        new_cnt = 0
        for data_index in range(0, fetched_total_data_count):
            target_image = rows[5][data_index]
            target_image_length = len(rows[1][data_index])
            if target_image_length <= self.digit_count:
                new_cnt += 1
                if self.add_margin:
                    if self.random_position_count > 0:
                        new_cnt += self.random_position_count
                    if self.random_rotate_count > 0:
                        new_cnt += self.random_rotate_count
        self.total_data_count = new_cnt
        self.update_data_counts()

        self.total_data = np.zeros((self.total_data_count, self.image_height, self.image_width, self.num_channels), dtype='uint8')
        self.total_label = np.zeros((self.total_data_count, self.digit_count + 1), dtype=int)
        print("total_data", self.total_data.shape)
        print("total_label", self.total_label.shape)

        print("start : {}".format(time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        total_data_index = 0
        for data_index in range(0, fetched_total_data_count):
            target_image = rows[5][data_index]
            target_image_length = len(rows[1][data_index])
            top = int(np.min(rows[3][data_index]))
            bottom = int(np.max(rows[0][data_index] + rows[3][data_index]))
            left = int(np.min(rows[2][data_index]))
            right = int(np.max(rows[2][data_index] + rows[4][data_index]))
            
            rand_max_width = 0
            rand_max_height = 0

            if self.add_margin:
                width = right - left
                height = bottom - top
                top = max(0, int(top - height * self.margin_ratio))
                bottom = min(len(target_image[0]), int(bottom + height * self.margin_ratio))
                left = max(0, int(left - width * self.margin_ratio))
                right = min(len(target_image[0][0]), int(right + width * self.margin_ratio))
                rand_max_width = int(width * self.margin_ratio)
                rand_max_height = int(height * self.margin_ratio)

            if target_image_length <= self.digit_count:
                #def generate_data(self, total_data_index, target_image_length, data_index, target_image,top,bottom,left,right):
                total_data_index = self.generate_data(rows,total_data_index,target_image_length,data_index,target_image,top,bottom,left,right)
                if self.add_margin:
                    if self.random_position_count > 0:
                        for rand_i in range(0, self.random_position_count):
                            rand_width = random.randrange(-rand_max_width, rand_max_width)
                            rand_height = random.randrange(-rand_max_height, rand_max_height)
                            rand_top = max(0, top + rand_height)
                            rand_bottom = min(len(target_image[0]), bottom + rand_height)
                            rand_left = max(0, left + rand_width)
                            rand_right = min(len(target_image[0][0]), right + rand_width)
                            
                            total_data_index = self.generate_data(rows,total_data_index,target_image_length,data_index,target_image,
                                                                  rand_top,rand_bottom,rand_left,rand_right)
                    if self.random_rotate_count > 0:
                        for rand_i in range(0, self.random_rotate_count):
                            total_data_index = self.generate_data(rows,total_data_index,target_image_length,data_index,target_image
                                                                  ,top,bottom,left,right, rotate_rand=True)
                    
#                 self.total_label[total_data_index][0] = target_image_length

#                 for digit_index in range(0, target_image_length):
#                     self.total_label[total_data_index][1 + digit_index] = rows[1][data_index][digit_index][0]
                
#                 for channel_index in range(0, self.num_channels):
#                     if self.use_standard_score:
#                         mean = np.mean(target_image[:][:][channel_index], dtype='float32')
#                         std = np.std(target_image[:][:][channel_index], dtype='float32', ddof=1)
#                         if std < 1e-4:std = 1
#                         target_image[:][:][channel_index] = ((target_image[:][:][channel_index] - mean) / std).astype('uint8')
                        
#                     resized_image = resize(target_image[channel_index][top:bottom, left:right],
#                                            output_shape=[self.image_width, self.image_height])
#                     for image_height_index in range(0, self.image_height):
#                         for image_width_index in range(0, self.image_width):
#                             self.total_data[total_data_index][image_height_index][image_width_index][channel_index] = \
#                                 resized_image[image_height_index][image_width_index] * 255

#                 if self.apply_gaussian_filter:
#                     self.total_data[total_data_index] = gaussian_filter(self.total_data[total_data_index], 1)
#                 total_data_index += 1

        print("end : {}".format(time.strftime("%Y-%m-%dT%H:%M:%S%z")))

        self.train_data = self.total_data[:self.train_data_count]
        self.train_label = self.total_label[:self.train_data_count]
        print("train_data", self.train_data.shape)
        print("train_label", self.train_label.shape)
        
        self.validation_data = self.total_data[self.train_data_count:self.train_data_count + self.validation_data_count]
        self.validation_label = self.total_label[
                                       self.train_data_count:self.train_data_count + self.validation_data_count]
        print("validation_data", self.validation_data.shape)
        print("validation_label", self.validation_label.shape)

        self.test_data = self.total_data[self.train_data_count + self.validation_data_count:]
        self.test_label = self.total_label[self.train_data_count + self.validation_data_count:]
        print("test_data", self.test_data.shape)
        print("test_label", self.test_label.shape)

