# pre-process.py

# should:
#   take the training (or test) directory containing movie reviews
#   perform pre-processing on each file
#   output the files in the vector format to be used by NB.py.

# Prior to building feature vectors:
#   separate punctuation from words
#   lowercase the words in the reviews.
#   use add-one smoothing for BOW features

# The training and the test files should have the following format:
#   one document per line
#   each line corresponds to a document
#   first column is the label
#   the other columns are feature values.

# Save the parameters of BOW model in a file called movie-review-BOW.NB .

import os
import sys


def count_frequencies(text):
    freq = {}
    for word in text:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq


def ignore_unseen_words(words, vocab):
    return [word for word in words if word in vocab]


def remove_punctuation(text):
    punctuation_to_remove = {'"', '*', '+', '.', '/', '<', '>', '@', '^', '_', '`', '{', '|', '~', ','}
    new_text = ""
    for char in text:
        if char is '!' or char is "?":
            new_text += " " + char
        elif char not in punctuation_to_remove:
            new_text += char.lower()
    return new_text.split()


def pretty_dict(dic):
    pretty = ""
    for key, val in dic.items():
        pretty += '"' + str(key) + '" : ' + str(val) + '\n'
    return pretty


def preprocess():
    feature_vectors = []
    for label in os.listdir(directory):                             # for each label
        folder = os.path.join(directory, label)
        if os.path.isdir(folder):
            for filename in os.listdir(folder):                     # for each document
                if filename.endswith(".txt"):
                    f = open(os.path.join(folder, filename), "r")
                    words = remove_punctuation(f.read())
                    words = ignore_unseen_words(words, vocab)
                    feature_vectors.append({label: count_frequencies(words)})

    name = "movie-review-BOW.NB - " + directory
    name = name.replace("/", " ")
    output = open(name, "w")
    for line in feature_vectors:
        output.write(pretty_dict(line))


directory = sys.argv[1]
vocab = set([line.rstrip() for line in open('all-reviews/imdb.vocab')])
preprocess()

