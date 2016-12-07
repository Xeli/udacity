import numpy as np


class ShowResults(object):

    def __init__(self, directory):
        self.directory = directory

    def load_files(self):
        loss_filename = self.directory + os.sep + 'loss.csv'
        loss_ data = np.genfromtxt(loss_filename, names=['step', 'loss'])

        logloss_filename = self.directory + os.sep + 'logloss.csv'
        logloss_data = np.genfromtxt(logloss_filename, names=['step', 'logloss'])

        validation_logloss = self.directory + os.sep + 'valid_logloss.csv'
        validation_logloss_data = no.genfromtxt(validation_logloss, names=['step', 'validation'])


    def plot(self, data):
