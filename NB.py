
# NB.py should take the following parameters:
#   the training file,
#   the test file,
#   the file where the parameters of the resulting model will be saved, and
#   the output file where you will write predictions made by the classifier on the test data (one example per line).

# The last line in the output file should list the overall accuracy of the classifier on the test data.


# def train_nb(documents, classes):
    # classes is a dictionary that holds key-label, value-# of documents with that label
    # we have vocab

import os
import sys
import json

training_file = sys.argv[1]  # 'movie-review-small.NB'
# test_file = sys.argv[2]
# model_output = sys.argv[3]
# predictions_output = sys.argv[4]


training_feature_vectors = []


def input_training_file():
    file = open(training_file, "r")
    for line in file.readlines():
        training_feature_vectors.append(json.loads(line))
    file.close()
#     
# def train_NB():
#     for document in training_feature_vectors:
#
#
