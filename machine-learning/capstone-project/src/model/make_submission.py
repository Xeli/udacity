from os import path, walk, sep
from test import getPredictions
import numpy as np


def get_filenames(directory):
    "Gather the filenames of all files inside this dir"
    filenames = []
    for root, dirs, files in walk(directory):
        absFiles = [root + sep + filename for filename in files]
        filenames = filenames + absFiles
    return filenames


def clean(i):
    return np.clip(i, 0.0001, 0.9999)

base_directory = path.dirname(path.abspath(__file__)) + '/../../'
dataset_dir = base_directory + "dataset/"

filenames = get_filenames(dataset_dir + "normalized_test_kaggle")

graphFile = base_directory + 'models/inception-v3/output_graph.pb'
labelFile = base_directory + 'models/inception-v3/output_labels.txt'
probabilities = getPredictions(graphFile, labelFile, filenames)

print("id, label")
for filename, prob in zip(filenames, probabilities):
    idOfFile = path.basename(filename).split('.')[0]
    print("{}, {}".format(idOfFile, clean(prob)))
