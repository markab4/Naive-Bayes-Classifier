# pre-process.py

# should:
#   take the training (or test) directory containing movie reviews
#   perform pre-processing on each file
#   output the files in the vector format to be used by NB.py.

# The training and the test files should have the following format:
#   one example per line
#   each line corresponds to an example
#   first column is the label
#   the other columns are feature values.

# Prior to building feature vectors:
#   separate punctuation from words
#   lowercase the words in the reviews.
#   use add-one smoothing for BOW features

# Save the parameters of BOW model in a file called movie-review-BOW.NB.

