# Movie Review Classification using Naïve Bayes

This repository implements in Python a Naïve Bayes classifier with bag-of-word (BOW ) features and Add-one smoothing. It implements the algorithm from scratch and does not use off-the-shelf software. 

## Scripts
In this repo, there are 2 scripts: NB.py and pre-process.py. 

NB.py takes the following parameters: 
- the training file
- the test file
- the file where the parameters of the resulting model will be saved
- the _output_ file which stores predictions made by the classifier on the test data (one document per line). 
   - The last line in the output file lists the overall accuracy of the classifier on the test data. 

pre-process.py:
- takes as a parameter the training (or test) directory containing movie reviews
- performs pre-processing on each file
   - Prior to building feature vectors, I separated punctuation from words and lowercase the words in the reviews.
- output the files in the vector format to be used by NB.py.

The training and the test files have the following format: 
- one example per line
   - each line corresponds to a document
- first column is the label
- the other columns are feature values.


## Test on Small Corpora

The following small corpus of movie reviews was used to initially train the classifier. The parameters of the model were saved in a file called movie-review-small.NB.

i. fun, couple, love, love **comedy**
ii. fast, furious, shoot **action**
iii. couple, fly, fast, fun, fun **comedy**
iv. furious, shoot, shoot, fun **action**
v. fly, fast, shoot, love **action** 

The classifier was tested on the new document below: 
_{fast, couple, shoot, fly}._ 
The most likely class was computed and the probabilities of each class are reported.

## Larger Movie Review Dataset

The movie review dataset provided in this repo was used to train a Naive Bayes classifier for the real task. I trained the classifier on the training data and tested it on the test data. 

The dataset contains movie reviews; each review is saved as a separate file in the folder “neg” or “pos” (which are located in “train” and “test” folders, respectively). I used these raw files and represented each review using a vector of bag-of-word features, where each feature corresponds to a word from the vocabulary file (also provided), and the value of the feature is the count of that word in the review file.

l trained the NB classifier on the training partition using the BOW features (using add-one smoothing) and evaluated the classifier on the test partition. In addition to BOW features, I experimented with a variation called binary multinomial Naive Bayes Classifier. A description of this is provided in the report. 

The parameters of the BOW model is saved in a file called movie-review-BOW.NB. 

My report also includes the accuracy of my program on the test data with BOW features and an investigation of my results, such as observed trends for the reviews for which my program made incorrect predictions.

