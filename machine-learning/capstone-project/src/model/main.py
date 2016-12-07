from os import path, walk, sep
from sklearn.model_selection import train_test_split
from test import testModel
from random import shuffle

def get_filenames(directory):
    "Gather the filenames of all files inside this dir"
    filenames = []
    for root, dirs, files in walk(directory):
        absFiles = [root + sep + filename for filename in files]
        filenames = filenames + absFiles
    return filenames

base_directory = path.dirname(path.abspath(__file__)) + '/../../'
dataset_dir = base_directory + "dataset/"

filenames = get_filenames(dataset_dir + "normalized_train")
shuffle(filenames)

labels = [path.basename(filename).split('.')[0] for filename in filenames]
X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.2)

graphFile = base_directory + 'models/inception-v3/output_graph.pb'
labelFile = base_directory + 'models/inception-v3/output_labels.txt'
logloss, percentageCorrect = testModel(graphFile, labelFile, X_test, y_test)

print("LogLoss: {}, Percentage correct: {}".format(logloss, percentageCorrect))
