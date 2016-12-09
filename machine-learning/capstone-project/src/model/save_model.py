import datetime
import os
import csv
from create_graph import CreateGraph
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter


class SaveModel(object):

    def __init__(self, base_dir, session, data, input_, output_):
        self.directory = self.create_directory(base_dir)
        self.session = session
        self.data = data
        self.input = input_
        self.output = output_

    def create_directory(self, basedir):
        now = datetime.datetime.now()
        files = len(os.listdir(basedir)) + 1
        directory_name = "{}-{}-{}-{}-{}".format(files, now.month, now.day, now.hour, now.minute)
        os.mkdir(basedir + os.sep + directory_name)
        return basedir + os.sep + directory_name + os.sep

    def write_data_to_file(self):
        for key in self.data:
            self.write_list_to_file(self.data[key], self.directory + key + '.csv')

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

cg = CreateGraph(2, 125, 1)

train_dir = dataset_dir + "normalized_train"
session, input_, output_, data = cg.train_model(train_dir, 16, 0.01, 4, 10000)

sm = SaveModel(base_directory + 'results', session, data, input_, output_)
sm.write_data_to_file()
sm.write_model_to_file()
