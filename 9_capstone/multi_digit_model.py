import numpy as np
import tensorflow as tf

from multi_digit_mnist_data import MultiDigitMNISTData


class MultiDigitModel(object):
    def __init__(self):
        self.num_channels = 1
        self.pooling_stride = 2

        self.num_steps = 10001
        self.batch_size = 128
        self.patch_size = 5
        self.depth_1 = 24
        self.depth_2 = 32
        self.depth_3 = self.depth_2
        self.depth_4 = self.depth_3
        self.depth_5 = self.depth_4
        self.depth_6 = self.depth_5
        self.depth_7 = self.depth_6
        self.depth_8 = self.depth_7 * 2

        self.num_hidden_1 = 1600
        self.num_hidden = 512

        self.beta = 0.001
        self.drop_out_rate = 0.5
        self.learning_rate_start = 0.001
        self.learning_rate_decay_steps = 1000000
        self.learning_rate_decay_rate = 0.96

        self.multi_digit_mnist = MultiDigitMNISTData(4)
        self.multi_digit_mnist.load_data()
        self.graph = tf.Graph()

    @staticmethod
    def accuracy_digit(p_length, p_digits,
                       batch_length_labels, batch_digits_labels, digit):
        eq_count = 0.0
        total_count = 0.0
        for i in range(0, len(p_digits[digit])):
            if np.argmax(batch_length_labels[i]) >= digit:
                total_count += 1.0
                if np.argmax(p_digits[digit][i]) == np.argmax(batch_digits_labels[i][digit]):
                    eq_count += 1.0
                    #         elif digit == 1:
                    #           print("np.argmax(p_digits[digit][i]):{}, np.argmax(batch_digits_labels[i][digit]:{}".format(np.argmax(p_digits[digit][i]), np.argmax(batch_digits_labels[i][digit])))

        if total_count == 0:
            return 0
        return eq_count / total_count * 100

    @staticmethod
    def accuracy_length(p_length, p_0, p_1, p_2, p_3, p_4,
                        batch_length_labels, batch_digits_labels):
        eq_count = 0.0
        for i in range(0, len(p_length)):
            if np.argmax(p_length[i]) == np.argmax(batch_length_labels[i]):
                eq_count += 1.0
        return eq_count / len(p_length) * 100

    @staticmethod
    def accuracy(p_length, p_digits,
                 batch_length_labels, batch_digits_labels):
        eq_count = 0.0
        for row_index in range(0, len(p_length)):
            # print("row_index:{}".format(row_index))
            one_based_length_predicted = np.argmax(p_length[row_index])
            one_based_length_real = np.argmax(batch_length_labels[row_index])

            # print("one_based_length_predicted : {}, one_based_length_real :{}".format(one_based_length_predicted, one_based_length_real))
            if one_based_length_predicted == one_based_length_real:
                is_equal = True
                for digit_index in range(0, one_based_length_real):
                    # print("\tdigit_index:{}".format(digit_index))
                    if np.argmax(p_digits[digit_index][row_index]) != np.argmax(
                            batch_digits_labels[row_index][digit_index]):
                        # print("\t\tnp.argmax(p_digits[digit_index][row_index]) :{}, np.argmax(batch_digits_labels[row_index][digit_index]) :{}".format(np.argmax(p_digits[digit_index][row_index]), np.argmax(batch_digits_labels[row_index][digit_index])))
                        is_equal = False
                        break;
                if is_equal:
                    eq_count += 1.0;

        return eq_count / len(p_length) * 100

    def run(self):
        with self.graph.as_default():
            # Input data.
            tf_train_dataset = tf.placeholder(tf.float32, shape=(
                self.batch_size, self.multi_digit_mnist.image_width, self.multi_digit_mnist.image_height,
                self.multi_digit_mnist.num_channels))
            print("tf_train_dataset : {}".format(tf_train_dataset))
            tf_train_length_labels = tf.placeholder(tf.float32,
                                                    shape=(self.batch_size, self.multi_digit_mnist.digit_count + 1))
            print("tf_train_length_labels : {}".format(tf_train_length_labels))
            tf_train_digits_labels = tf.placeholder(tf.float32,
                                                    shape=(self.batch_size, self.multi_digit_mnist.digit_count, 10))
            print("tf_train_digits_labels : {}".format(tf_train_digits_labels))
            tf_valid_dataset = tf.constant(self.multi_digit_mnist.validation_data)
            tf_test_dataset = tf.constant(self.multi_digit_mnist.test_data)

            # Variables.
            layer1_weights = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.num_channels, self.depth_1], stddev=0.1))
            layer1_biases = tf.Variable(tf.truncated_normal([self.depth_1], stddev=0.1))
            layer2_weights = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.depth_1, self.depth_2], stddev=0.1))
            layer2_biases = tf.Variable(tf.truncated_normal([self.depth_2], stddev=0.1))
            layer3_weights = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.depth_2, self.depth_3], stddev=0.1))
            layer3_biases = tf.Variable(tf.truncated_normal([self.depth_3], stddev=0.1))
            layer4_weights = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.depth_3, self.depth_4], stddev=0.1))
            layer4_biases = tf.Variable(tf.truncated_normal([self.depth_4], stddev=0.1))
            layer5_weights = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.depth_4, self.depth_5], stddev=0.1))
            layer5_biases = tf.Variable(tf.truncated_normal([self.depth_5], stddev=0.1))
            layer6_weights = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.depth_5, self.depth_6], stddev=0.1))
            layer6_biases = tf.Variable(tf.truncated_normal([self.depth_6], stddev=0.1))
            layer7_weights = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.depth_6, self.depth_7], stddev=0.1))
            layer7_biases = tf.Variable(tf.truncated_normal([self.depth_7], stddev=0.1))
            layer8_weights = tf.Variable(
                tf.truncated_normal([self.patch_size, self.patch_size, self.depth_7, self.depth_8], stddev=0.1))
            layer8_biases = tf.Variable(tf.truncated_normal([self.depth_8], stddev=0.1))

            flatted = int(round(float(self.multi_digit_mnist.image_width) / (self.pooling_stride ** 2)) * round(
                float(self.multi_digit_mnist.image_height) / (self.pooling_stride ** 2)) * self.depth_8)
            fc_layer1_weights = tf.Variable(tf.truncated_normal([flatted, self.num_hidden_1], stddev=0.1))
            fc_layer1_biases = tf.Variable(tf.truncated_normal([self.num_hidden_1], stddev=0.1))
            fc_layer2_weights = tf.Variable(tf.truncated_normal([self.num_hidden_1, self.num_hidden], stddev=0.1))
            fc_layer2_biases = tf.Variable(tf.truncated_normal([self.num_hidden], stddev=0.1))

            digit_length_weights = tf.Variable(
                tf.truncated_normal([self.num_hidden, self.multi_digit_mnist.digit_count + 1], stddev=0.1))
            digit_length_biases = tf.Variable(tf.truncated_normal([self.multi_digit_mnist.digit_count + 1], stddev=0.1))
            digits_0_weights = tf.Variable(tf.truncated_normal([self.num_hidden, 10], stddev=0.1))
            digits_0_biases = tf.Variable(tf.truncated_normal([10], stddev=0.1))
            digits_1_weights = tf.Variable(tf.truncated_normal([self.num_hidden, 10], stddev=0.1))
            digits_1_biases = tf.Variable(tf.truncated_normal([10], stddev=0.1))
            digits_2_weights = tf.Variable(tf.truncated_normal([self.num_hidden, 10], stddev=0.1))
            digits_2_biases = tf.Variable(tf.truncated_normal([10], stddev=0.1))
            digits_3_weights = tf.Variable(tf.truncated_normal([self.num_hidden, 10], stddev=0.1))
            digits_3_biases = tf.Variable(tf.truncated_normal([10], stddev=0.1))
            digits_4_weights = tf.Variable(tf.truncated_normal([self.num_hidden, 10], stddev=0.1))
            digits_4_biases = tf.Variable(tf.truncated_normal([10], stddev=0.1))

            # Model.
            def model(data, is_training=True):
                stride = 2
                conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.relu(conv + layer1_biases)
                conv = tf.nn.max_pool(conv, [1, stride, stride, 1], [1, stride, stride, 1], padding='SAME')
                conv = tf.nn.conv2d(conv, layer2_weights, [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.relu(conv + layer2_biases)
                conv = tf.nn.max_pool(conv, [1, stride, stride, 1], [1, stride, stride, 1], padding='SAME')
                conv = tf.nn.conv2d(conv, layer3_weights, [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.relu(conv + layer3_biases)
                #     conv = tf.nn.conv2d(conv, layer4_weights, [1, 1, 1, 1], padding='SAME')
                #     conv = tf.nn.relu(conv + layer4_biases)
                #     conv = tf.nn.conv2d(conv, layer5_weights, [1, 1, 1, 1], padding='SAME')
                #     conv = tf.nn.relu(conv + layer5_biases)
                #     conv = tf.nn.conv2d(conv, layer6_weights, [1, 1, 1, 1], padding='SAME')
                #     conv = tf.nn.relu(conv + layer6_biases)
                #     conv = tf.nn.conv2d(conv, layer7_weights, [1, 1, 1, 1], padding='SAME')
                #     conv = tf.nn.relu(conv + layer7_biases)
                conv = tf.nn.conv2d(conv, layer8_weights, [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.relu(conv + layer8_biases)
                if is_training:
                    conv = tf.nn.dropout(conv, self.drop_out_rate)

                shape = conv.get_shape().as_list()
                reshape = tf.reshape(conv, [shape[0], shape[1] * shape[2] * shape[3]])
                reshape = tf.nn.relu(tf.matmul(reshape, fc_layer1_weights) + fc_layer1_biases)
                reshape = tf.nn.relu(tf.matmul(reshape, fc_layer2_weights) + fc_layer2_biases)

                digit_length_logit = tf.matmul(reshape, digit_length_weights) + digit_length_biases
                digits_0_logit = tf.matmul(reshape, digits_0_weights) + digits_0_biases
                digits_1_logit = tf.matmul(reshape, digits_1_weights) + digits_1_biases
                digits_2_logit = tf.matmul(reshape, digits_2_weights) + digits_2_biases
                digits_3_logit = tf.matmul(reshape, digits_3_weights) + digits_3_biases
                digits_4_logit = tf.matmul(reshape, digits_4_weights) + digits_4_biases

                return digit_length_logit, digits_0_logit, digits_1_logit, digits_2_logit, digits_3_logit, digits_4_logit

            # Training computation.
            logits = model(tf_train_dataset, True)
            #   digits_1_mult = tf.to_float((tf.argmax(tf.nn.softmax(tf_train_length_labels), dimension=1) > 1))
            #   digits_2_mult = tf.to_float((tf.argmax(tf.nn.softmax(tf_train_length_labels), dimension=1) > 2))
            #   digits_3_mult = tf.to_float((tf.argmax(tf.nn.softmax(tf_train_length_labels), dimension=1) > 3))
            #   digits_4_mult = tf.to_float((tf.argmax(tf.nn.softmax(tf_train_length_labels), dimension=1) > 4))
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits[0], tf_train_length_labels)
                + tf.nn.softmax_cross_entropy_with_logits(logits[1], tf_train_digits_labels[:, 0])
                + tf.nn.softmax_cross_entropy_with_logits(logits[2], tf_train_digits_labels[:, 1])
                + tf.nn.softmax_cross_entropy_with_logits(logits[3], tf_train_digits_labels[:, 2])
                + tf.nn.softmax_cross_entropy_with_logits(logits[4], tf_train_digits_labels[:, 3])
                #         + tf.nn.softmax_cross_entropy_with_logits(logits[5], tf_train_digits_labels[:,4])
                #         + beta * tf.nn.l2_loss(layer1_weights)
                #         + beta * tf.nn.l2_loss(layer2_weights)
                #         + beta * tf.nn.l2_loss(layer3_weights)
                #         + beta * tf.nn.l2_loss(layer4_weights)
                #         + beta * tf.nn.l2_loss(layer5_weights)
                #         + beta * tf.nn.l2_loss(layer6_weights)
                #         + beta * tf.nn.l2_loss(layer7_weights)
                #         + beta * tf.nn.l2_loss(layer8_weights)
                #         + beta * tf.nn.l2_loss(fc_layer1_weights)
                #         + beta * tf.nn.l2_loss(fc_layer2_weights)
                #         + beta * tf.nn.l2_loss(digit_length_weights)
                #         + beta * tf.nn.l2_loss(digits_0_weights)
                #         + beta * tf.nn.l2_loss(digits_1_weights)
                #         + beta * tf.nn.l2_loss(digits_2_weights)
                #         + beta * tf.nn.l2_loss(digits_3_weights)
                #         + beta * tf.nn.l2_loss(digits_4_weights)
            )

            # Optimizer.
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate_start, global_step,
                                                       self.learning_rate_decay_steps,
                                                       self.learning_rate_decay_rate, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            # Predictions for the training, validation, and test data.
            train_length_prediction = tf.nn.softmax(logits[0])
            train_digits_0_prediction = tf.nn.softmax(logits[1])
            train_digits_1_prediction = tf.nn.softmax(logits[2])
            train_digits_2_prediction = tf.nn.softmax(logits[3])
            train_digits_3_prediction = tf.nn.softmax(logits[4])
            train_digits_4_prediction = tf.nn.softmax(logits[5])

        import time

        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            print('Initialized')
            for step in range(self.num_steps):
                offset = (step * self.batch_size) % (self.multi_digit_mnist.train_data.shape[0] - self.batch_size)
                batch_data = self.multi_digit_mnist.train_data[offset:(offset + self.batch_size), :, :, :]
                batch_length_labels = self.multi_digit_mnist.train_label_length[offset:(offset + self.batch_size), 0, :]
                batch_digits_labels = self.multi_digit_mnist.train_label_digits[offset:(offset + self.batch_size), :]

                feed_dict = {tf_train_dataset: batch_data, tf_train_length_labels: batch_length_labels,
                             tf_train_digits_labels: batch_digits_labels}
                _, l, p_length, p_0, p_1, p_2, p_3, p_4 = session.run(
                    [optimizer, loss, train_length_prediction, train_digits_0_prediction, train_digits_1_prediction,
                     train_digits_2_prediction, train_digits_3_prediction, train_digits_4_prediction],
                    feed_dict=feed_dict)
                if step % 100 == 0:
                    print('Minibatch loss at step %d: %f' % (step, l))
                    accuracy_result = self.accuracy_length(p_length, p_0, p_1, p_2, p_3, p_4, batch_length_labels,
                                                           batch_digits_labels)
                    print('Minibatch accuracy_length: %.1f%%' % accuracy_result)
                    for k in range(0, 4):
                        accuracy_result = self.accuracy_digit(p_length, [p_0, p_1, p_2, p_3, p_4], batch_length_labels,
                                                              batch_digits_labels, k)
                        print("Minibatch accuracy_digit_{}".format(k) + ": %.1f%%" % accuracy_result)
                    accuracy_result = self.accuracy(p_length, [p_0, p_1, p_2, p_3, p_4], batch_length_labels,
                                                    batch_digits_labels)
                    print('Minibatch accuracy: %.1f%%' % accuracy_result)
                    print("finish : {}".format(time.strftime("%Y-%m-%dT%H:%M:%S%z")))
