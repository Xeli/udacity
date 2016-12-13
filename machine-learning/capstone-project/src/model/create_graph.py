import tensorflow as tf
from os import path, walk, sep
import numpy as np
from scipy import ndimage


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

    def simple_add_layer(self, layer, filters, biases, padding):
        layer = tf.nn.conv2d(layer, filters, strides=[1, 1, 1, 1], padding=padding)
        layer = tf.nn.relu(layer + biases)

        return layer

    def add_layer(self, layer, f1, b1, f3, b3, f5, b5, f1p, b1p, padding, dropout=False):
        print(layer.get_shape())
        print(f1.get_shape())
        layer1 = tf.nn.conv2d(layer, f1, strides=[1, 1, 1, 1], padding=padding)
        layer1 = tf.nn.relu(layer1 + b1)

        layer3 = tf.nn.conv2d(layer, f3, strides=[1, 1, 1, 1], padding=padding)
        layer3 = tf.nn.relu(layer3 + b3)

        layer5 = tf.nn.conv2d(layer, f5, strides=[1, 1, 1, 1], padding=padding)
        layer5 = tf.nn.relu(layer5 + b5)

        pool = tf.nn.avg_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        pool = tf.nn.conv2d(pool, f1p, strides=[1, 1, 1, 1], padding=padding)
        pool = tf.nn.relu(pool + b1p)

        layer = tf.concat(3, [layer1, layer3, layer5, pool])

        if dropout:
            layer = tf.nn.dropout(layer, 0.75)

        return layer

    def train_model(self, train, validation, test, param):
        batch_size = param['batch_size']
        layers = param['layers']
        hidden_nodes = param['hidden_nodes']
        dropout_input = param['dropout_input']
        dropout_hidden = param['dropout_hidden_layers']
        learning_mode = param['learning_mode']
        learning_rate = param['learning_rate']
        steps = param['steps']

        filter_count = 16
        padding = 'SAME'

        X_train, y_train = train
        X_valid, y_valid = validation
        X_valid = self.get_data(X_valid)
        X_test, y_test = test

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

        train = tf_train_dataset
        if dropout_input:
            train = tf.nn.dropout(train, dropout_input)

        valid = tf_validation_dataset
        test = tf_test_dataset

        filter_count_adj = self.image_channels
        for i in range(0, layers):
            f1, b1 = self.create_weights(1, filter_count_adj, filter_count)
            f3, b3 = self.create_weights(3, filter_count_adj, filter_count)
            f5, b5 = self.create_weights(5, filter_count_adj, filter_count)
            f1p, b1p = self.create_weights(1, filter_count_adj, filter_count)

            train = self.add_layer(train, f1, b1, f3, b3, f5, b5, f1p, b1p, padding, dropout_hidden)
            valid = self.add_layer(valid, f1, b1, f3, b3, f5, b5, f1p, b1p, padding)
            test = self.add_layer(test, f1, b1, f3, b3, f5, b5, f1p, b1p, padding)

            train_shape = train.get_shape()
            if i > 1 and train_shape[1] > 4 and i % 2 == 0:
                ksize = [1, 2, 2, 1]
                strides = [1, 2, 2, 1]
                train = tf.nn.avg_pool(train, ksize=ksize, strides=strides, padding='SAME')
                valid = tf.nn.max_pool(valid, ksize=ksize, strides=strides, padding='SAME')
                test = tf.nn.max_pool(test, ksize=ksize, strides=strides, padding='SAME')

            filter_count_adj = filter_count * 4

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
            learning_rate,
            batch * batch_size,
            len(X_train),
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
            'validation-accuracy': [],
            'validation-logloss': []
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
                accuracy, logloss = self.test(predictions, batch_labels)
                print('Minibatch accuracy: %.1f%%' % accuracy)
                print('Minibatch logloss: {}'.format(logloss))

                data['accuracy'].append(accuracy)
                data['logloss'].append(logloss)

                feed_dict = {
                    tf_validation_dataset: X_valid
                }
                predictions = session.run(validation_prediction, feed_dict=feed_dict)
                accuracy, logloss = self.test(predictions, y_valid)
                print('Validation accuracy: %.1f%%' % accuracy)
                print('Validation logloss: {}'.format(logloss))
                data['validation-accuracy'].append(accuracy)
                data['validation-logloss'].append(logloss)

        def f(filename):
            shape = (1, self.image_size, self.image_size, self.image_channels)
            image_data = self.get_data([filename])[0].reshape(shape)
            feed_dict = {
                tf_test_dataset: image_data
            }
            return session.run(test_prediction, feed_dict=feed_dict)

        predictions = [f(data) for data in X_test]
        predictions = np.array(predictions).reshape(len(predictions), self.label_count)

        accuracy, logloss = self.test(predictions, y_test)
        print('Testset accuracy: {}'.format(accuracy))
        print('Testset logloss: {}'.format(logloss))

        return session, tf_test_dataset, test_prediction, data

    def test(self, predictions, labels):
        return self.accuracy(predictions, labels), self.logloss(predictions, labels)

    def accuracy(self, predictions, labels):
        total = predictions.shape[0]
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / total

    def logloss(self, predictions, labels):
        error = 0
        for prediction, label in zip(predictions.tolist(), labels.tolist()):
            error += label[0] * np.log(prediction[0]) + (1-label[1])*np.log(1 - prediction[1])

        return (-1.0/(len(predictions))) * error

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
