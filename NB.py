
# NB.py should take the following parameters:
#   the training file,
#   the test file,
#   the file where the parameters of the resulting model will be saved, and
#   the output file where you will write predictions made by the classifier on the test data (one example per line).

# The last line in the output file should list the overall accuracy of the classifier on the test data.


import sys
import json
import math


def get_inputs():
    training = sys.argv[1]
    test = sys.argv[2]
    model_output_file = sys.argv[3]
    predictions_output_file = sys.argv[4]
    vocab = set([line.rstrip() for line in open('all-reviews/imdb.vocab')])
    documents = []
    classes = {}
    test_docs = {}

    # training file
    training_file = open(training, "r")

    for line in training_file.readlines():
        vector = json.loads(line)
        documents.append(vector)
        label = list(vector.keys())[0]
        if label in classes:
            classes[label].append(vector[label])
        else:
            classes[label] = [vector[label]]
    # The following code replaces the previous loop if using binary NB
    # for line in training_file.readlines():
    #     vector = json.loads(line)
    #     documents.append(vector)
    #     label = list(vector.keys())[0]
    #     clipped = {word: 1 if count > 0 else 0 for word, count in vector[label].items()}
    #     if label in classes:
    #         classes[label].append(clipped)
    #     else:
    #         classes[label] = [clipped]
    training_file.close()

    # test file
    test_file = open(test, "r")

    for line in test_file.readlines():
        vector = json.loads(line)
        label = list(vector.keys())[0]
        if label in test_docs:
            test_docs[label].append(bow_to_list(vector[label]))
        else:
            test_docs[label] = [bow_to_list(vector[label])]
    # # The following code replaces the previous loop if using binary NB
    # for line in test_file.readlines():
    #     vector = json.loads(line)
    #     label = list(vector.keys())[0]
    #     if label in test_docs:
    #         test_docs[label].append(vector[label].keys())
    #     else:
    #         test_docs[label] = [vector[label].keys()]
    test_file.close()

    return documents, classes, vocab, test_docs, model_output_file, predictions_output_file


def train_nb(documents, classes, vocab):
    total_num_of_documents = len(documents)
    log_prior = {}
    bow_for_each_class = {}
    log_likelihood = {}
    num_of_words_in_each_class = {}
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
    return log_prior, log_likelihood, bow_for_each_class


def arg_max(d):
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def test_nb(test_doc, classes, vocab, log_prior, log_likelihood):
    sum_of_log_probs = {}
    for label, docs_in_the_class in classes.items():
        sum_of_log_probs[label] = log_prior[label]
        for word in test_doc:
            if word in vocab:
                sum_of_log_probs[label] += log_likelihood[(word, label)]
        # print("probability of class", label, "is", sum_of_log_probs[label])
    return arg_max(sum_of_log_probs)


def answer_questions():
    documents, classes, vocab, test_docs, model_output, predictions_output = get_inputs()
    log_prior, log_likelihood, bow_in_each_class = train_nb(documents, classes, vocab)
    results = {True: 0, False: 0}
    predictions = "Document # \t Predicted Label \t Actual Label\n"
    num = 1
    for label, documents in test_docs.items():
        for document in documents:
            test_result = test_nb(document, classes, vocab, log_prior, log_likelihood)
            results[test_result == label] += 1
            predictions += "\t" + str(num) + "\t\t | \t\t" + test_result + "\t\t | \t\t" + label + "\n"
            num += 1
    model_output_file = open(model_output, "w")
    model = "Log prior probability of each class:\n" + str(log_prior) + \
            '\n\nLog likelihood of each word: \n' + pretty_prob(log_likelihood)
    model_output_file.write(model)
    model_output_file.close()
    predictions_output_file = open(predictions_output, "w")
    accuracy = results[True] / (results[False] + results[True]) * 100
    predictions += "Total: " + str(results) + ". Accuracy: " + str(accuracy) + '%'
    predictions_output_file.write(predictions)
    predictions_output_file.close()


def pretty_prob(dic):
    pretty = ""
    for key, val in dic.items():
        w = str(key[0])
        c = str(key[1])
        pretty += 'P(' + w + ' | ' + c + ') = ' + str(val) + '\n'
    return pretty


def bow_to_list(bow):
    output = []
    for word, freq in bow.items():
        for i in range(freq):
            output.append(word)
    return output


answer_questions()
