import numpy as np


class MultiDigitMNISTData(object):
    def __init__(self, digit_count=1):
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')

        self.mnist = mnist
        self.digit_count = digit_count
        self.image_width = 28
        self.image_height = 28 * self.digit_count
        self.target_digit_num_labels = 10
        self.num_channels = 1

        self.total_data_count = len(mnist.data)
        self.train_data_count = int(0.9 * self.total_data_count)
        self.validation_data_count = int(0.05 * self.total_data_count)
        self.test_data_count = self.total_data_count - self.train_data_count - self.validation_data_count

        self.multi_digit_mnist_data = np.zeros([self.total_data_count, self.image_width * self.image_height],
                                               dtype=np.float32)
        self.multi_digit_mnist_target_length = np.zeros([self.total_data_count, 1],
                                                        dtype=np.float32)
        self.multi_digit_mnist_target_digits = np.zeros([self.total_data_count, self.digit_count],
                                                        dtype=np.float32)

        self.train_data = None
        self.train_label_length = None
        self.train_label_digits = None
        self.validation_data = None
        self.validation_label_length = None
        self.validation_label_digits = None
        self.test_data = None
        self.test_label_length = None
        self.test_label_digits = None

    def reformat_dataset(self, dataset):
        return dataset.reshape((-1, self.image_width, self.image_height, self.num_channels)).astype(np.float32)

    def reformat_target_length(self, target_length):
        return (np.arange(self.digit_count + 1) == target_length[:, None]).astype(np.float32)

    def reformat_target_digits(self, target_digits):
        return (np.arange(self.target_digit_num_labels) == target_digits[:, :, None]).astype(np.float32)

    def load_data(self):
        import random
        for i in range(0, self.total_data_count):
            random_length = random.randrange(1, self.digit_count + 1)
            self.multi_digit_mnist_target_length[i, 0] = random_length
            for j in range(0, random_length):
                random_index = random.randrange(0, len(self.mnist.data))
                self.multi_digit_mnist_data[i, j * 784:(j + 1) * 784] = self.mnist.data[random_index] / 255.0
                self.multi_digit_mnist_target_digits[i, j] = self.mnist.target[random_index]
        print("multi_digit_mnist_data", self.multi_digit_mnist_data.shape)  # (70000, 3920)
        print("multi_digit_mnist_target_length", self.multi_digit_mnist_target_length.shape)  # (70000, 1)
        print("multi_digit_mnist_target_digits", self.multi_digit_mnist_target_digits.shape)  # (70000, 5)

        self.train_data = self.reformat_dataset(self.multi_digit_mnist_data[:self.train_data_count])
        self.train_label_length = self.reformat_target_length(
            self.multi_digit_mnist_target_length[:self.train_data_count])
        self.train_label_digits = self.reformat_target_digits(
            self.multi_digit_mnist_target_digits[:self.train_data_count])
        print("train_data", self.train_data.shape)  # (49000, 140, 28, 1)
        print("train_label_length", self.train_label_length.shape)  # (49000, 1, 6)
        print("train_label_digits", self.train_label_digits.shape)  # (49000, 5, 10)

        self.validation_data = self.reformat_dataset(
            self.multi_digit_mnist_data[self.train_data_count:self.train_data_count + self.validation_data_count])
        self.validation_label_length = self.reformat_target_length(self.multi_digit_mnist_target_length[
                                                                   self.train_data_count:self.train_data_count + self.validation_data_count])
        self.validation_label_digits = self.reformat_target_digits(self.multi_digit_mnist_target_digits[
                                                                   self.train_data_count:self.train_data_count + self.validation_data_count])
        print("validation_data", self.validation_data.shape)  # (14000, 140, 28, 1)
        print("validation_label_length", self.validation_label_length.shape)  # (14000, 1, 6)
        print("validation_label_digits", self.validation_label_digits.shape)  # (14000, 5, 10)

        self.test_data = self.reformat_dataset(
            self.multi_digit_mnist_data[self.train_data_count + self.validation_data_count:])
        self.test_label_length = self.reformat_target_length(
            self.multi_digit_mnist_target_length[self.train_data_count + self.validation_data_count:])
        self.test_label_digits = self.reformat_target_digits(
            self.multi_digit_mnist_target_digits[self.train_data_count + self.validation_data_count:])
        print("test_data", self.test_data.shape)  # (7000, 140, 28, 1)
        print("test_label_length", self.test_label_length.shape)  # (7000, 1, 6)
        print("test_label_digits", self.test_label_digits.shape)  # (7000, 5, 10)
