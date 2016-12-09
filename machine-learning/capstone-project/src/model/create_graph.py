import tensorflow as tf
from os import path, walk, sep
from random import shuffle
import numpy as np
from scipy import ndimage
from cross_version import train_test_split


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

    def add_layer(self, layer, filter_, biases, padding, dropout=False, pool=False, k=2):
        if dropout:
            layer = tf.nn.dropout(layer, 0.75)

        layer = tf.nn.conv2d(layer, filter_, strides=[1, 1, 1, 1], padding=padding)
        layer = tf.nn.relu(layer + biases)

        if pool:
            layer = tf.nn.max_pool(layer, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

        return layer

    def train_model(self, dataset_dir, batch_size, initial_learning_rate, layers, steps):
        filter_count = 16
        hidden_nodes = 64
        padding = 'SAME'

        filenames = self.get_filenames(dataset_dir)
        shuffle(filenames)

        labels = self.get_labels(filenames)

        X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.1)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.01)

        X_valid = self.get_data(X_valid)

        print("Training data size: {}".format(len(X_train)))
        print("Validation data size: {}".format(len(X_valid)))
        print("Test data size: {}".format(len(X_test)))

        # Input data.
        shape = (batch_size, self.image_size, self.image_size, self.image_channels)
        tf_train_dataset = tf.placeholder(tf.float32, shape=shape)

        shape = (len(X_valid), self.image_size, self.image_size, self.image_channels)
        tf_validation_dataset = tf.placeholder(tf.float32, shape=shape)
        shape = (1, self.image_size, self.image_size, self.image_channels)
        tf_test_dataset = tf.placeholder(tf.float32, shape=shape)

        filters, biases = self.create_weights(5, self.image_channels, filter_count)
        train = self.add_layer(tf_train_dataset, filters, biases, padding, dropout=True, pool=True)
        valid = self.add_layer(tf_validation_dataset, filters, biases, padding, pool=True)
        test = self.add_layer(tf_test_dataset, filters, biases, padding, pool=True)

        for i in range(0, layers):
            filters, biases = self.create_weights(5, filter_count, filter_count)
            train = self.add_layer(train, filters, biases, padding, dropout=True)
            valid = self.add_layer(valid, filters, biases, padding)
            test = self.add_layer(test, filters, biases, padding)

            train = self.add_layer(train, filters, biases, padding, dropout=True, pool=True)
            valid = self.add_layer(valid, filters, biases, padding, pool=True)
            test = self.add_layer(test, filters, biases, padding, pool=True)

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

        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            batch * batch_size,
            len(filenames),
            0.95,
            staircase=True
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = optimizer.minimize(loss, global_step=batch)

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
        if tf.__version__ >= '0.12.0':
            session.run(tf.global_variables_initializer())
        else:
            session.run(tf.initialize_all_variables())

        for step in range(steps):
            offset = (step * batch_size) % (len(y_train) - batch_size)
            batch_filenames = X_train[offset:(offset + batch_size)]
            batch_data = self.get_data(batch_filenames)
            batch_labels = y_train[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels: batch_labels}
            args = [optimizer, loss, train_prediction]
            _, l, predictions = session.run(args, feed_dict=feed_dict)
            if (step % 150 == 0):
                print('Step: {}'.format(step))
                print('Minibatch loss at step %d: %f' % (step, l))
                acc = self.accuracy(predictions, batch_labels)
                print('Minibatch accuracy: %.1f%%' % acc)
                feed_dict = {
                    tf_validation_dataset: X_valid
                }
                predictions = session.run(validation_prediction, feed_dict=feed_dict)
                validation = self.accuracy(predictions, y_valid)
                print('Validation accuracy: %.1f%%' % validation)
                data['loss'].append(l)
                data['accuracy'].append(acc)
                data['validation'].append(validation)

        def f(filename):
            shape = (1, self.image_size, self.image_size, self.image_channels)
            image_data = self.get_data([filename])[0].reshape(shape)
            feed_dict = {
                tf_test_dataset: image_data
            }
            return session.run(test_prediction, feed_dict=feed_dict)

        predictions = [f(data) for data in X_test]
        predictions = np.array(predictions).reshape(len(predictions), self.label_count)
        print('Testset accuracy: {}'.format(self.accuracy(predictions, y_test)))

        return session, tf_test_dataset, test_prediction, data

    def accuracy(self, predictions, labels):
        total = predictions.shape[0]
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / total

    def get_data(self, filenames):
        shape = (len(filenames), self.image_size, self.image_size, self.image_channels)
        dataset = np.ndarray(shape=shape, dtype=np.float32)
        for i, filename in enumerate(filenames):
            data = ((ndimage.imread(filename)).astype(float) - 255.0 / 2) / 255.0
            data = data.reshape((self.image_size, self.image_size, 1))
            dataset[i, :, :] = data

        return self.reformat(dataset)

    def get_labels(self, filenames):
        labels = [path.basename(filename).split('.')[0] for filename in filenames]
        labels = [0 if label == 'dog' else 1 for label in labels]
        labels = np.array(labels, dtype=np.int32)
        labels = self.dense_to_one_hot(labels, 2)
        return labels

    def get_filenames(self, directory):
        "Gather the filenames of all files inside this dir"
        filenames = []
        for root, dirs, files in walk(directory):
            absFiles = [root + sep + filename for filename in files]
            filenames = filenames + absFiles
        return filenames

    def reformat(self, dataset):
        dataset = dataset.reshape(
          (-1, self.image_size, self.image_size, self.image_channels)).astype(np.float32)
        return dataset

    def dense_to_one_hot(self, labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
