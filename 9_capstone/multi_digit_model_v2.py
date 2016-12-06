import re

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat



class MultiDigitModel(object):
    def __init__(self, multi_digit_dataset, digit_count=1, num_channels=1, pooling_stride=2, num_steps=10001,
                 batch_size=128, patch_size=5, num_convs=None, num_fc_1=3072, num_fc_2=3072, beta=0.001,
                 use_drop_out=False, drop_out_rate=0.5, use_norm=False, use_LCN=False,
                 learning_rate_start=0.1, learning_rate_decay_rate=0.1, add_l2_loss=False):
        if num_convs is None:
            # The number of units at each spatial location in each layer is [48, 64, 128, 160] for the first four layers and 192 for all other locally connected layers.
            num_convs = [48, 64, 128, 160, 192, 192, 192, 192]

        # The fully connected layers contain 3,072 units each.
        self.num_channels = num_channels
        self.pooling_stride = pooling_stride

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_conv_1 = num_convs[0]
        self.num_conv_2 = num_convs[1]
        self.num_conv_3 = num_convs[2]
        self.num_conv_4 = num_convs[3]
        self.num_conv_5 = num_convs[4]
        self.num_conv_6 = num_convs[5]
        self.num_conv_7 = num_convs[6]
        self.num_conv_8 = num_convs[7]

        self.last_num_conv = 0
        for i in range(7, -1, -1):
            if num_convs[i] > 0:
                self.last_num_conv = num_convs[i]
                break

        self.num_fc_1 = num_fc_1
        self.num_fc_2 = num_fc_2

        self.beta = beta
        self.learning_rate_start = learning_rate_start
        self.learning_rate_decay_rate = learning_rate_decay_rate

        self.digit_count = digit_count
        self.multi_digit_dataset = multi_digit_dataset
        self.graph = tf.Graph()

        self.add_l2_loss = add_l2_loss
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
        self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

        # Constants describing the training process.
        self.MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
        self.NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.

        self.use_norm = use_norm
        self.use_drop_out = use_drop_out
        self.drop_out_rate = drop_out_rate
        self.use_LCN = use_LCN


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
                        break
                if is_equal:
                    eq_count += 1.0

        return eq_count / len(p_length) * 100, eq_count, len(p_length)


    def _activation_summary(self, x):
        tf.histogram_summary(x.op.name + '/activations', x)
        tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

    def _add_loss_summaries(self, total_loss):
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name +' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        return loss_averages_op


    def _variable_on_cpu(self, name, shape, initializer):
        return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def conv_layer(self, number, input_data, input_num, output_num):
        with tf.variable_scope('conv_' + str(number)) as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=[self.patch_size, self.patch_size, input_num, output_num],
                                                      stddev=5e-2,
                                                      wd=0.0)
            conv = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [output_num], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv)
        if self.use_norm:
            conv = tf.nn.lrn(conv, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm' + str(number))
        return conv

    # inference.
    def inference(self, data, is_training=True):
        stride = 2
        conv = None
        with tf.variable_scope('conv_1') as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=[self.patch_size, self.patch_size, self.num_channels, self.num_conv_1],
                                                      stddev=5e-2,
                                                      wd=0.0)
            conv = tf.nn.conv2d(data, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [self.num_conv_1], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            # TODO change to max out unit.
            # The first hidden layer contains maxout units (Goodfellow et al., 2013) (with three filters per unit)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv1)
        # TODO stride varies over conv net. 1,2.
        # The max pooling window size is 2 × 2. The stride alternates between 2 and 1 at each layer, so that half of the layers don’t reduce the spatial size of the representation
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, stride, stride, 1], padding='SAME', name='pool1')

        if self.use_norm:
            # TODO change normalize parameters
            # The subtractive normalization operates on 3x3 windows and preserves representation size.
            conv = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        else:
            conv = pool1

        with tf.variable_scope('conv_2') as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=[self.patch_size, self.patch_size, self.num_conv_1, self.num_conv_2],
                                                      stddev=5e-2,
                                                      wd=0.0)
            conv = tf.nn.conv2d(conv, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [self.num_conv_2], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv2)
        if self.use_norm:
            conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, stride, stride, 1], padding='SAME', name='pool2')
        conv = pool2

        if self.num_conv_3 > 0:
            conv = self.conv_layer(3, conv, self.num_conv_2, self.num_conv_3)
        if self.num_conv_4 > 0:
            conv = self.conv_layer(4, conv, self.num_conv_3, self.num_conv_4)
        if self.num_conv_5 > 0:
            conv = self.conv_layer(5, conv, self.num_conv_4, self.num_conv_5)
        if self.num_conv_6 > 0:
            conv = self.conv_layer(6, conv, self.num_conv_5, self.num_conv_6)
        if self.num_conv_7 > 0:
            conv = self.conv_layer(7, conv, self.num_conv_6, self.num_conv_7)
        if self.num_conv_8 > 0:
            conv = self.conv_layer(8, conv, self.num_conv_7, self.num_conv_8)

        if self.use_drop_out:
            conv = tf.nn.dropout(conv, self.drop_out_rate)

        with tf.variable_scope('local_1') as scope:
            shape = conv.get_shape().as_list()
            reshape = tf.reshape(conv, [shape[0], -1])
            dim = reshape.get_shape()[1].value
            weights = self._variable_with_weight_decay('weights', shape=[dim, self.num_fc_1],stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [self.num_fc_1], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            self._activation_summary(local3)

        with tf.variable_scope('local_2') as scope:
            weights = self._variable_with_weight_decay('weights', shape=[self.num_fc_1, self.num_fc_2],stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [self.num_fc_2], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
            self._activation_summary(local4)

        with tf.variable_scope('digit_length') as scope:
            weights = self._variable_with_weight_decay('weights', [self.num_fc_2, self.multi_digit_dataset.digit_count + 1],stddev=1.0/self.num_fc_2, wd=0.0)
            biases = self._variable_on_cpu('biases', [self.multi_digit_dataset.digit_count + 1],tf.constant_initializer(0.0))
            digit_length_logit = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            self._activation_summary(digit_length_logit)

        with tf.variable_scope('digit_0') as scope:
            weights = self._variable_with_weight_decay('weights', [self.num_fc_2, self.multi_digit_dataset.target_digit_num_labels],stddev=1.0/self.num_fc_2, wd=0.0)
            biases = self._variable_on_cpu('biases', [self.multi_digit_dataset.target_digit_num_labels],tf.constant_initializer(0.0))
            digits_0_logit = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            self._activation_summary(digits_0_logit)
        with tf.variable_scope('digit_1') as scope:
            weights = self._variable_with_weight_decay('weights', [self.num_fc_2, self.multi_digit_dataset.target_digit_num_labels],stddev=1.0/self.num_fc_2, wd=0.0)
            biases = self._variable_on_cpu('biases', [self.multi_digit_dataset.target_digit_num_labels],tf.constant_initializer(0.0))
            digits_1_logit = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            self._activation_summary(digits_1_logit)
        with tf.variable_scope('digit_2') as scope:
            weights = self._variable_with_weight_decay('weights', [self.num_fc_2, self.multi_digit_dataset.target_digit_num_labels],stddev=1.0/self.num_fc_2, wd=0.0)
            biases = self._variable_on_cpu('biases', [self.multi_digit_dataset.target_digit_num_labels],tf.constant_initializer(0.0))
            digits_2_logit = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            self._activation_summary(digits_2_logit)
        with tf.variable_scope('digit_3') as scope:
            weights = self._variable_with_weight_decay('weights', [self.num_fc_2, self.multi_digit_dataset.target_digit_num_labels],stddev=1.0/self.num_fc_2, wd=0.0)
            biases = self._variable_on_cpu('biases', [self.multi_digit_dataset.target_digit_num_labels],tf.constant_initializer(0.0))
            digits_3_logit = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            self._activation_summary(digits_3_logit)
        with tf.variable_scope('digit_4') as scope:
            weights = self._variable_with_weight_decay('weights', [self.num_fc_2, self.multi_digit_dataset.target_digit_num_labels],stddev=1.0/self.num_fc_2, wd=0.0)
            biases = self._variable_on_cpu('biases', [self.multi_digit_dataset.target_digit_num_labels],tf.constant_initializer(0.0))
            digits_4_logit = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            self._activation_summary(digits_4_logit)

        return digit_length_logit, digits_0_logit, digits_1_logit, digits_2_logit, digits_3_logit, digits_4_logit


    def loss_(self, logits, length_label, digit_labels):
        # Calculate the average cross entropy loss across the batch.
        length_label = tf.cast(length_label, tf.int64)
        digit_labels = tf.cast(digit_labels, tf.int64)
        #         print(logits[0], length_label, digit_labels)
        cross_entropy_sum = tf.nn.softmax_cross_entropy_with_logits(logits[0], length_label, name='cross_entropy_for_length')
        for i in range(0, self.digit_count):
            #             print(i, logits[i + 1], digit_labels[:,i])
            cross_entropy_sum += tf.nn.softmax_cross_entropy_with_logits(logits[i + 1], digit_labels[:, i], name='cross_entropy_for_digit' + str(i))
        cross_entropy_mean = tf.reduce_mean(cross_entropy_sum, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')


    def train(self, total_loss, global_step):
        # Variables that affect learning rate.
        num_batches_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.learning_rate_start,
                                        global_step,
                                        decay_steps,
                                        self.learning_rate_decay_rate,
                                        staircase=True)
        tf.scalar_summary('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            #opt = tf.train.GradientDescentOptimizer(lr)
            opt = tf.train.AdagradOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def run(self):
        import time

        with self.graph.as_default():
            # Input data.
            tf_train_dataset = tf.placeholder(tf.float32, shape=(
                self.batch_size, self.multi_digit_dataset.image_width, self.multi_digit_dataset.image_height,
                self.multi_digit_dataset.num_channels), name="tf_train_dataset")
            print("tf_train_dataset : {}".format(tf_train_dataset))
            tf_train_length_labels = tf.placeholder(tf.float32,
                                                    shape=(self.batch_size, self.multi_digit_dataset.digit_count + 1))
            print("tf_train_length_labels : {}".format(tf_train_length_labels))
            tf_train_digits_labels = tf.placeholder(tf.float32,
                                                    shape=(self.batch_size, self.multi_digit_dataset.digit_count, 11))
            print("tf_train_digits_labels : {}".format(tf_train_digits_labels))

            # Training computation.
            logits = self.inference(tf_train_dataset, True)
            #   digits_1_mult = tf.to_float((tf.argmax(tf.nn.softmax(tf_train_length_labels), dimension=1) > 1))
            #   digits_2_mult = tf.to_float((tf.argmax(tf.nn.softmax(tf_train_length_labels), dimension=1) > 2))
            #   digits_3_mult = tf.to_float((tf.argmax(tf.nn.softmax(tf_train_length_labels), dimension=1) > 3))
            #   digits_4_mult = tf.to_float((tf.argmax(tf.nn.softmax(tf_train_length_labels), dimension=1) > 4))
            loss = self.loss_(logits, tf_train_length_labels, tf_train_digits_labels)

            global_step = tf.Variable(0, trainable=False)
            train_op = self.train(loss, global_step)

            # Predictions for the training, validation, and test data.
            train_length_prediction = tf.nn.softmax(logits[0], name='p_digit_length')
            train_digits_0_prediction = tf.nn.softmax(logits[1], name='p_digit_0')
            train_digits_1_prediction = tf.nn.softmax(logits[2], name='p_digit_1')
            train_digits_2_prediction = tf.nn.softmax(logits[3], name='p_digit_2')
            train_digits_3_prediction = tf.nn.softmax(logits[4], name='p_digit_3')
            train_digits_4_prediction = tf.nn.softmax(logits[5], name='p_digit_4')

            summary_op = tf.merge_all_summaries()
            init = tf.initialize_all_variables()
            session = tf.Session()
            session.run(init)
            summary_writer = tf.train.SummaryWriter('logs/tensorboard/voyageth/svhn/v2/' + time.strftime("%Y-%m-%dT%H:%M:%S%z"), session.graph)

            saver = tf.train.Saver()

            print('Initialized')
            for step in range(self.num_steps):
                offset = (step * self.batch_size) % (self.multi_digit_dataset.train_data.shape[0] - self.batch_size)
                batch_data = self.multi_digit_dataset.train_data[offset:(offset + self.batch_size), :, :, :]
                batch_length_labels = self.multi_digit_dataset.reformat_target_length(self.multi_digit_dataset.train_label[offset:(offset + self.batch_size), 0, None])
                batch_digits_labels = self.multi_digit_dataset.reformat_target_digits(self.multi_digit_dataset.train_label[offset:(offset + self.batch_size), 1:, None])

                feed_dict = {tf_train_dataset: batch_data, tf_train_length_labels: batch_length_labels,
                             tf_train_digits_labels: batch_digits_labels}

                _, summary_str, l, p_length, p_0, p_1, p_2, p_3, p_4 = session.run([train_op, summary_op, loss, train_length_prediction, train_digits_0_prediction, train_digits_1_prediction,
                                                                                    train_digits_2_prediction, train_digits_3_prediction, train_digits_4_prediction],feed_dict=feed_dict)
                if step % 100 == 0:
                    summary_writer.add_summary(summary_str, step)

                if step % 500 == 0:
                    print('Minibatch loss at step %d: %f' % (step, l))
                    accuracy_result = self.accuracy_length(p_length, p_0, p_1, p_2, p_3, p_4, batch_length_labels,
                                                           batch_digits_labels)
                    print('Minibatch accuracy_length: %.1f%%' % accuracy_result)
                    for k in range(0, self.digit_count):
                        accuracy_result = self.accuracy_digit(p_length, [p_0, p_1, p_2, p_3, p_4], batch_length_labels,
                                                              batch_digits_labels, k)
                        print("Minibatch accuracy_digit_{}".format(k) + ": %.1f%%" % accuracy_result)
                    accuracy_result, _, _ = self.accuracy(p_length, [p_0, p_1, p_2, p_3, p_4], batch_length_labels,
                                                          batch_digits_labels)
                    print('Minibatch accuracy: %.1f%%' % accuracy_result)

                    validation_accuracy_numerator = 0
                    validation_accuracy_denominator = 0
                    for step in range(0, len(self.multi_digit_dataset.validation_label) / self.batch_size):
                        valid_offset = (step * self.batch_size) % (self.multi_digit_dataset.validation_data.shape[0] - self.batch_size)
                        valid_batch_data = self.multi_digit_dataset.validation_data[valid_offset:(valid_offset + self.batch_size), :, :, :]
                        valid_batch_length_labels = self.multi_digit_dataset.reformat_target_length(self.multi_digit_dataset.validation_label[valid_offset:(valid_offset + self.batch_size), 0, None])
                        valid_batch_digits_labels = self.multi_digit_dataset.reformat_target_digits(self.multi_digit_dataset.validation_label[valid_offset:(valid_offset + self.batch_size), 1:, None])

                        feed_dict = {tf_train_dataset: valid_batch_data, tf_train_length_labels: valid_batch_length_labels,
                                     tf_train_digits_labels: valid_batch_digits_labels}

                        p_length, p_0, p_1, p_2, p_3, p_4 = session.run([train_length_prediction,
                                                                         train_digits_0_prediction,
                                                                         train_digits_1_prediction,
                                                                         train_digits_2_prediction,
                                                                         train_digits_3_prediction,
                                                                         train_digits_4_prediction],feed_dict=feed_dict)

                        accuracy_result, numerator, denominator = self.accuracy(p_length, [p_0, p_1, p_2, p_3, p_4],
                                                                                valid_batch_length_labels, valid_batch_digits_labels)
                        validation_accuracy_numerator += numerator
                        validation_accuracy_denominator += denominator
                    accuracy_result = validation_accuracy_numerator / validation_accuracy_denominator * 100
                    print('Validation accuracy: %.1f%%' % accuracy_result)
                    print("finish : {}".format(time.strftime("%Y-%m-%dT%H:%M:%S%z")))

            test_accuracy_numerator = 0
            test_accuracy_denominator = 0
            for step in range(0, len(self.multi_digit_dataset.test_label) / self.batch_size):
                test_offset = (step * self.batch_size) % (self.multi_digit_dataset.test_data.shape[0] - self.batch_size)
                test_batch_data = self.multi_digit_dataset.test_data[test_offset:(test_offset + self.batch_size), :, :, :]
                test_batch_length_labels = self.multi_digit_dataset.reformat_target_length(self.multi_digit_dataset.test_label[test_offset:(test_offset + self.batch_size), 0, None])
                test_batch_digits_labels = self.multi_digit_dataset.reformat_target_digits(self.multi_digit_dataset.test_label[test_offset:(test_offset + self.batch_size), 1:, None])

                feed_dict = {tf_train_dataset: test_batch_data, tf_train_length_labels: test_batch_length_labels,
                             tf_train_digits_labels: test_batch_digits_labels}

                p_length, p_0, p_1, p_2, p_3, p_4 = session.run([train_length_prediction,
                                                                 train_digits_0_prediction,
                                                                 train_digits_1_prediction,
                                                                 train_digits_2_prediction,
                                                                 train_digits_3_prediction,
                                                                 train_digits_4_prediction],feed_dict=feed_dict)

                accuracy_result, numerator, denominator = self.accuracy(p_length, [p_0, p_1, p_2, p_3, p_4],
                                                                        test_batch_length_labels, test_batch_digits_labels)
                test_accuracy_numerator += numerator
                test_accuracy_denominator += denominator
            accuracy_result = test_accuracy_numerator / test_accuracy_denominator * 100
            print('Test accuracy: %.1f%%' % accuracy_result)
            finish_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            print("finish : {}".format(finish_time))
            #             path = saver.save(session, "{}_MultiDigitNN.pb".format(time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            #             print("Model saved. path : %s" % path)

            output_graph_def = graph_util.convert_variables_to_constants(
                session, self.graph.as_graph_def(), ['p_digit_length', 'p_digit_0', 'p_digit_1', 'p_digit_2', 'p_digit_3', 'p_digit_4'])
            with gfile.FastGFile('./graph/v2/model_{}.pb'.format(finish_time), 'wb') as f:
                f.write(output_graph_def.SerializeToString())
                        

