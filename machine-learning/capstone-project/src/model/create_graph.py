import tensorflow as tf
from os import path, walk, sep
from random import shuffle
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split


class CreateGraph(object):
    "Helper class to create tensorflow graph"
    def __init__(self, label_count, image_size, image_channels):
        self.label_count = label_count
        self.image_size = image_size
        self.image_channels = image_channels
        self.stddev = 0.1

    def create_weights(self, conv_size, channels, filters):
        shape = [conv_size, conv_size, channels, filters]

        filter_ = tf.Variable(tf.truncated_normal(shape, stddev=self.stddev))
        biases = tf.Variable(tf.zeros([filters]))

        return filter_, biases

    def add_layer(self, filter_, biases, padding, prev):
        strides = [1, 2, 2, 1]
        conv = tf.nn.conv2d(prev, filter_, strides=strides, padding=padding)
        hidden = tf.nn.relu(conv + biases)

        return hidden

    def train_model(self, dataset_dir, batch_size, learning_rate, layers):
        filter_count = 16
        hidden_nodes = 64

        dataset, labels = self.get_data(dataset_dir, self.image_channels)

        dataset, X_test, labels, y_test = train_test_split(dataset, labels, test_size=0.1)
        X_train, X_valid, y_train, y_valid = train_test_split(dataset, labels, test_size=0.1)

        # Input data.
        shape = (batch_size, self.image_size, self.image_size, self.image_channels)
        tf_train_dataset = tf.placeholder(tf.float32, shape=shape)

        tf_validation_dataset = tf.constant(X_valid)
        shape = (1, self.image_size, self.image_size, self.image_channels)
        tf_test_dataset = tf.placeholder(tf.float32, shape=shape)

        filters, biases = self.create_weights(3, self.image_channels, filter_count)
        train = self.add_layer(filters, biases, 'SAME', tf_train_dataset)
        valid = self.add_layer(filters, biases, 'SAME', tf_validation_dataset)
        test = self.add_layer(filters, biases, 'SAME', tf_test_dataset)
        for i in range(0, layers):
            filters, biases = self.create_weights(3, filter_count, filter_count)
            train = self.add_layer(filters, biases, 'SAME', train)
            valid = self.add_layer(filters, biases, 'SAME', valid)
            test = self.add_layer(filters, biases, 'SAME', test)

        # create fully connected output layer
        shape = train.get_shape().as_list()
        connected_weights = tf.Variable(tf.truncated_normal(
            [shape[1] * shape[2] * shape[3], hidden_nodes], stddev=self.stddev))
        connected_biases = tf.Variable(tf.constant(0.1, shape=[hidden_nodes]))

        output_weights = tf.Variable(tf.truncated_normal(
            [hidden_nodes, self.label_count], stddev=self.stddev))
        output_biases = tf.Variable(tf.constant(1.0, shape=[self.label_count]))

        shape = train.get_shape().as_list()
        new_shape = [shape[0], shape[1] * shape[2] * shape[3]]
        reshape = tf.reshape(train, new_shape)
        connected = tf.nn.relu(tf.matmul(reshape, connected_weights) + connected_biases)
        model_train = tf.matmul(connected, output_weights) + output_biases

        shape = valid.get_shape().as_list()
        new_shape = [shape[0], shape[1] * shape[2] * shape[3]]
        reshape = tf.reshape(valid, new_shape)
        connected = tf.nn.relu(tf.matmul(reshape, connected_weights) + connected_biases)
        model_valid = tf.matmul(connected, output_weights) + output_biases

        shape = test.get_shape().as_list()
        new_shape = [shape[0], shape[1] * shape[2] * shape[3]]
        reshape = tf.reshape(test, new_shape)
        connected = tf.nn.relu(tf.matmul(reshape, connected_weights) + connected_biases)
        model_test = tf.matmul(connected, output_weights) + output_biases

        label_shape = (batch_size, self.label_count)
        tf_train_labels = tf.placeholder(tf.float32, shape=label_shape)
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(model_train, tf_train_labels))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        train_prediction = tf.nn.softmax(model_train)
        validation_prediction = tf.nn.softmax(model_valid)
        test_prediction = tf.nn.softmax(model_test)

        data = {
            'accuracy': [],
            'logloss': [],
            'loss': [],
            'validation': []
        }

        session = tf.InteractiveSession()
        session.run(tf.global_variables_initializer())

        for step in range(1500):
            print('Step: {}'.format(step))
            offset = (step * batch_size) % (len(y_train) - batch_size)
            batch_data = X_train[offset:(offset + batch_size), :, :, :]
            batch_labels = y_train[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels: batch_labels}
            args = [optimizer, loss, train_prediction]
            _, l, predictions = session.run(args, feed_dict=feed_dict)
            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                acc = self.accuracy(predictions, batch_labels)
                print('Minibatch accuracy: %.1f%%' % acc)
                validation = self.accuracy(validation_prediction.eval(), y_valid)
                print('Validation accuracy: %.1f%%' % validation)
                data['loss'].append(l)
                data['accuracy'].append(acc)
                data['validation'].append(validation)
        return session, tf_test_dataset, test_prediction, data

    def accuracy(self, predictions, labels):
        total = predictions.shape[0]
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / total

    def get_data(self, directory, image_channels):
        filenames = self.get_filenames(directory)[:500]
        shuffle(filenames)

        shape = (len(filenames), self.image_size, self.image_size, image_channels)
        dataset = np.ndarray(shape=shape, dtype=np.float32)
        for i, filename in enumerate(filenames):
            dataset[i, :, :] = ((ndimage.imread(filename)).astype(float) - 255.0 / 2) / 255.0

        labels = [path.basename(filename).split('.')[0] for filename in filenames]
        labels = [0 if label == 'dog' else 1 for label in labels]
        labels = np.array(labels, dtype=np.int32)
        labels = self.dense_to_one_hot(labels, 2)
        return self.reformat(image_channels, dataset), labels

    def get_filenames(self, directory):
        "Gather the filenames of all files inside this dir"
        filenames = []
        for root, dirs, files in walk(directory):
            absFiles = [root + sep + filename for filename in files]
            filenames = filenames + absFiles
        return filenames

    def reformat(self, image_channels, dataset):
        dataset = dataset.reshape(
          (-1, self.image_size, self.image_size, image_channels)).astype(np.float32)
        return dataset

    def dense_to_one_hot(self, labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
