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
import re

directory = sys.argv[1]
pos_reviews = os.path.join(directory, "pos")
neg_reviews = os.path.join(directory, "neg")

for folder in [pos_reviews, neg_reviews]:
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            f = open(os.path.join(folder, filename), "r")
            text = f.read()
            review = re.findall(r"[\w']+|[.,!?;:]", text)
            print(text)
