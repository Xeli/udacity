from os import path, listdir
import operator
import csv
import json


def get_csv_data(csv_file):
    result = {}
    with open(csv_file, mode='r') as in_file:
        reader = csv.reader(in_file)
        for rows in reader:
            result[int(rows[0])] = rows[1]
    return result


def process_result_dir(directory):
    files = listdir(directory)
    csv_files = [directory + '/' + f for f in files if f.endswith('.csv')]
    data = {}
    for csv_file in csv_files:
        name = path.splitext(path.basename(csv_file))[0]
        data[name] = get_csv_data(csv_file)
        if name == 'test-logloss' or name == 'test-accuracy':
            data[name] = data[name][0]
    with open(directory + '/config.json') as config_file:
        data['config'] = json.load(config_file)
    return data


def count_config_frequencies(placeholder, data):
    counts = {}
    for key in placeholder:
        counts[key] = {}

    for row in data:
        for key in row['config']:
            value = row['config'][key]
            if value not in counts[key]:
                counts[key][value] = 0
            counts[key][value] += 1
    return counts

base_directory = path.dirname(path.abspath(__file__)) + '/../../'

results_dir = base_directory + 'results/'
results = listdir(results_dir)
results = [results_dir + f for f in results if path.isdir(results_dir + f)]
data = [process_result_dir(d) for d in results]

data = sorted(data, key=lambda x: x['test-logloss'])
data = data[9:]

size = int(len(data)*0.113)
size = 7
trimmed_data = data[:size]

counts = count_config_frequencies(trimmed_data[0]['config'], trimmed_data)

for key in counts:
    sorted_x = sorted(counts[key].items(), key=operator.itemgetter(1))
    print(key)
    print(sorted_x)

print([x['test-logloss'] for x in trimmed_data])
