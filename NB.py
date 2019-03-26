
# NB.py should take the following parameters:
#   the training file,
#   the test file,
#   the file where the parameters of the resulting model will be saved, and
#   the output file where you will write predictions made by the classifier on the test data (one example per line).

# The last line in the output file should list the overall accuracy of the classifier on the test data.


import os
import sys
import json
import math

training_file = sys.argv[1]  # 'movie-review-small.NB'
# test_file = sys.argv[2]
# model_output = sys.argv[3]
# predictions_output = sys.argv[4]


documents = []
classes = {}
log_prior = {}
vocab = set([line.rstrip() for line in open('all-reviews/imdb.vocab')])
bow_for_each_class = {}
num_of_words_in_each_class = {}
log_likelihood = {}


def input_training_file():
    file = open(training_file, "r")
    for line in file.readlines():
        vector = json.loads(line)
        documents.append(vector)
        key = list(vector.keys())[0]
        if key in classes:
            classes[key].append(vector[key])
        else:
            classes[key] = [vector[key]]
    file.close()


def train_nb():
    total_num_of_documents = len(documents)
    for label, docs_in_the_class in classes.items():
        # calculate P(c) terms
        num_of_documents_in_this_class = len(docs_in_the_class)
        log_prior[label] = math.log2(num_of_documents_in_this_class / total_num_of_documents)
        bow_for_each_class[label] = {}
        num_of_words_in_each_class[label] = 0
        for doc in docs_in_the_class:
            for word, value in doc.items():
                num_of_words_in_each_class[label] += value
                if word in bow_for_each_class[label]:
                    bow_for_each_class[label][word] += value
                else:
                    bow_for_each_class[label][word] = value

        # calculate P(w|c) terms
        for word in vocab:
            count = 0
            if word in bow_for_each_class[label]:
                count = bow_for_each_class[label][word]
            log_likelihood[(word, label)] = math.log2(
                (count + 1) /
                (num_of_words_in_each_class[label] + len(vocab)))

    # print("documents", documents,
    #       "\nclasses", classes,
    #       "\nlogprior", log_prior,
    #       "\nbigdoc", bow_for_each_class,
    #       "\ncount", num_of_words_in_each_class,
    #       "\nlog likelihood", log_likelihood,
    #       "\nlength of vocab", len(vocab)
    #       )


input_training_file()
train_nb()
