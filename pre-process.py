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
#   one example per line
#   each line corresponds to an example
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
    for i in range(len(words)):
        if words[i] not in vocab:
            words.pop(i)


def preprocess():
    directory = sys.argv[1]

    punctuation_to_remove = {'"', '*', '+', '.', '/', '<', '>', '@', '^', '_', '`', '{', '|', '~', ','}

    vocab = set([line.rstrip() for line in open('all-reviews/imdb.vocab')])

    BOW = {}
    classes = {}

    for label in os.listdir(directory):                             # for each label
        classes[label] = 0
        folder = os.path.join(directory, label)
        if os.path.isdir(folder):
            concat_text = ""
            for filename in os.listdir(folder):                     # for each document
                if filename.endswith(".txt"):
                    classes[label] += 1
                    f = open(os.path.join(folder, filename), "r")
                    text = f.read()
                    for char in text:
                        if char is '!' or char is "?":
                            concat_text += " " + char
                        elif char not in punctuation_to_remove:
                            concat_text += char.lower()
                    concat_text += " "
            words = concat_text.split()
            ignore_unseen_words(words, vocab)
            BOW[label] = count_frequencies(words)
    print(BOW)
    print(classes)


preprocess()

