import os
import sys
from unzip import unzip
from convert import convert

base_directory = os.path.dirname(os.path.abspath(__file__)) + '/../../'
dataset_dir = base_directory + "dataset/"

if len(sys.argv) != 3:
    print("Provide path to train and test set.")
    sys.exit()

trainzip = sys.argv[1]
testzip = sys.argv[2]
if not (os.path.exists(trainzip) and os.path.exists(testzip)):
    print("Could not find the training or test zip file.")
    sys.exit()


print("Unzipping..")
unzip(trainzip, dataset_dir)
unzip(testzip, dataset_dir)

print("Normalizing..")
convert(dataset_dir + "train", dataset_dir + "normalized_train", 500)
convert(dataset_dir + "test", dataset_dir + "normalized_test", 125)
