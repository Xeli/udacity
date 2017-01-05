import csv
import matplotlib.pyplot as plt
import sys

path = sys.argv[1]

data = {
    'train': [],
    'validation': []
}

with open(path + '/train-logloss.csv') as csvfile:
    train_logloss = csv.reader(csvfile)
    for row in train_logloss:
        data['train'].append(float(row[1]))

with open(path + '/validation-logloss.csv') as csvfile:
    validation_logloss = csv.reader(csvfile)
    for row in validation_logloss:
        data['validation'].append(float(row[1]))

print(data)
averaged_data = {
    'train': [],
    'validation': []
}

ave_size = 10
for key in averaged_data:
    for i in range(len(data[key]) // ave_size):
        index = i * ave_size
        ave = data[key][index]
        for offset in range(ave_size):
            ave = data[key][index + offset]
        ave /= ave_size
        averaged_data[key].append(ave)

data = averaged_data

offset_size = 50 * ave_size
t = plt.plot(range(0, len(data['train']) * offset_size, offset_size), data['train'], label='Training')
v = plt.plot(range(0, len(data['validation']) * offset_size, offset_size), data['validation'], color='r', label='Validation')
plt.ylabel('LogLoss')
plt.legend()
plt.show()
