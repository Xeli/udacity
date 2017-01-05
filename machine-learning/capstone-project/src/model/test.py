from __future__ import print_function
import numpy as np
import tensorflow as tf


def create_graph(modelPath):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def testModel(modelPath, labelsPath, filenames, labels):
    create_graph(modelPath)

    f = open(labelsPath, 'r')
    lines = f.readlines()
    labelOrder = [w.strip() for w in lines]
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        error = 0
        correctAnswers = 0
        n = len(filenames)
        for i, (filename, label) in enumerate(zip(filenames, labels)):
            print('Classified {} / {}'.format(i, n), end='\r')
            image_data = tf.gfile.FastGFile(filename, 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            result = {}
            for i, l in enumerate(labelOrder):
                result[l] = predictions[i]

            error += (-1.0/n) * logloss(label, result)
            correctAnswers += correct(label, result)
        correctAnswers = correctAnswers / float(n)
    return (error, correctAnswers)


def getPredictions(modelPath, labelsPath, filenames):
    create_graph(modelPath)
    f = open(labelsPath, 'r')
    lines = f.readlines()
    labelOrder = [w.strip() for w in lines]

    answers = []
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        n = len(filenames)
        for i, filename in enumerate(filenames):
            print('Classified {} / {}'.format(i, n), end='\r')
            image_data = tf.gfile.FastGFile(filename, 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            result = {}
            for i, l in enumerate(labelOrder):
                result[l] = predictions[i]
            answers.append(result['dog'])
    return answers


def logloss(label, result):
    y = 1 if label == 'dog' else 0
    return y * np.log(result['dog']) + (1-y)*np.log(1 - result['dog'])


def correct(label, result):
    y = 1 if label == 'dog' else 0
    if np.round(result['dog']) == y:
        return 1
    return 0
