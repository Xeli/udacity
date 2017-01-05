import datetime
import os
import csv
import json
from create_graph import CreateGraph
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from collections import OrderedDict
from random import shuffle
from cross_version import train_test_split
from copy import deepcopy
import numpy as np


class SaveModel(object):

    def __init__(self, base_dir, session, data, input_, output_, config):
        self.directory = self.create_directory(base_dir)
        self.session = session
        self.data = data
        self.input = input_
        self.output = output_
        self.config = config

    def save(self):
        self.write_config_to_file()
        self.write_data_to_file()

    def create_directory(self, basedir):
        now = datetime.datetime.now()
        files = len(os.listdir(basedir)) + 1
        directory_name = "{}-{}-{}-{}-{}".format(files, now.month, now.day, now.hour, now.minute)
        os.mkdir(basedir + os.sep + directory_name)
        return basedir + os.sep + directory_name + os.sep

    def write_config_to_file(self):
        with open(self.directory + 'config.json', 'w') as fp:
            json.dump(self.config, fp)

    def write_data_to_file(self):
        for key in self.data:
            self.write_list_to_file(self.data[key], self.directory + key + '.csv')

    def write_submission(self, data):
        new_file = open(self.directory + 'submission.csv')
        write = csv.write(new_file, quoting=csv.QUOTE_ALL)
        writer.writerows(data)

    def write_list_to_file(self, data, filename):
        new_file = open(filename, 'w')
        writer = csv.writer(new_file, quoting=csv.QUOTE_ALL)
        new_data = enumerate(data)
        writer.writerows(new_data)

    def write_model_to_file(self):
        saver = tf.train.Saver(sharded=True)
        model_exporter = exporter.Exporter(saver)
        model_exporter.init(
            self.session.graph.as_graph_def(),
            named_graph_signatures={
                'inputs': exporter.generic_signature({'images': self.input}),
                'outputs': exporter.generic_signature({'scores': self.output})
            })
        model_exporter.export(self.directory, tf.constant('1'), self.session)

base_directory = os.path.dirname(os.path.abspath(__file__)) + '/../../'
dataset_dir = base_directory + "dataset/"

cg = CreateGraph(label_count=2, image_size=500, image_channels=1)

# get training and validation set
train_dir = dataset_dir + "normalized_train"
filenames = cg.get_filenames(train_dir)
shuffle(filenames)
labels = cg.get_labels(filenames)
X_train, X_valid, y_train, y_valid = train_test_split(filenames, labels, test_size=0.001)
train = (X_train, y_train)
validation = (X_valid, y_valid)

# get test set
test_dir = dataset_dir + "normalized_test"
X_test = cg.get_filenames(test_dir)
y_test = cg.get_labels(X_test)
test = (X_test, y_test)

# parameters
parameters = {
    'batch_size': [32],
    'layers': [4, 8, 12],
    'hidden_nodes': [32, 64],
    'dropout_input': [False, 0.75],
    'dropout_hidden_layers': [False, 0.5],
    'learning_rate': [0.05, 0.1, 'dynamic'],
    'filter_count': [32],
    'epochs': [1, 10],
}
parameters = OrderedDict(parameters.items())

# generate a list of all possible parameter combinations
params = [{}]
while len(parameters) > 0:
    key, values = parameters.popitem()
    new_params = []
    for value in values:
        for new_param in deepcopy(params):
            new_param[key] = value
            new_params.append(new_param)
    params = new_params

# for param in params:
param = {
    'batch_size': 32,
    'layers': 8,
    'hidden_nodes': 16,
    'dropout_input': False,
    'dropout_hidden_layers': False,
    'learning_rate': 0.05,
    'filter_count': 32,
    'epochs': 1,
}

# params = [param]

for param in params:
    session, input_, output_, data = cg.train_model(train, validation, test, param)

    sm = SaveModel(base_directory + 'results', session, data, input_, output_, param)
    sm.save()

    filenames = cg.get_filenames(dataset_dir + 'normalized_test_kaggle')
    predictions = cg.run_dataset(session, output_, input_, filenames)

    data = [('id', 'label')]
    for filename, prob in zip(filenames, predictions):
        idOfFile = os.path.basename(filename).split('.')[0]
        prob = np.clip(prob, 1E-7, 1-1E-7)
        data.append((idOfFile, prob))
    sm.write_submission(data)
    cg.close(session)
