#!python3

'''

voteclf

just the classes for vote clf and perceptron

perceptron takes nn as input, and just cleans it's output and such

voteclf takes multiple models as input

'''

# other personal scripts
from WordLexicon import *

# general use scripts
import pickle
import numpy as np
from collections import Counter

# task specific scripts
import keras
import tensorflow as tf

from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier as sk
from sklearn.linear_model import LogisticRegression, SGDClassifier

# settings imports

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# define classes

class VoteClf:
    def __init__(self, bool_clfs=None, onehot_clfs=None, lexicon=None, conf_thresh=1):
        self.bool_clfs = bool_clfs
        self.onehot_clfs = onehot_clfs
        self.lexicon = lexicon
        self.vocab_count = len(lexicon.wordset)
        self.conf_thresh = conf_thresh

    def create_bool_tables(self, feature_set):

        if self.lexicon == None:
            print('require a lexicon to use bool_classifiers\n\n')
            assert self.lexicon != None

        bool_table = {}
        for word in self.lexicon.wordset:
            bool_table[word] = (self.lexicon.forward[word] in feature_set)

        return bool_table

    def encode_onehot_data(self, feature_set):
        
        vec = np.zeros(self.vocab_count)
        for datum in feature_set:
            vec[datum] = 1

        return vec

    def flatten_bool_table(self, table):
        pass

    def classify(self, features):
        # features are intaken as....index sets
        # convert from index to bool and onehot as needed
        votes = []
        
        if self.bool_clfs is not None:
            bool_table = self.create_bool_tables(features)
            #
            for clf in self.bool_clfs:
                votes.append(clf.classify(bool_table))

            del(bool_table)

        if self.onehot_clfs is not None:
            vector_data = self.encode_onehot_data(features)
            #
            for clf in self.onehot_clfs:
                votes.append(clf.classify(vector_data))

            del(vector_data)

        result = Counter(votes).most_common(1)
        outcome = result[0][0]
        confidence = result[0][1] / len(votes)

        votes.append(outcome)

        return outcome, confidence, votes

    def test(self, data_set):

        correct = 0
        conf_correct = 0

        total = 0
        conf_total = 0

        feature_set = [a for (a, b) in data_set]
        label_set = [b for (a, b) in data_set]

        for features in feature_set:
            result, conf, votes = self.classify(features)

            if result == label_set[total]:
                correct += 1

                if conf >= self.conf_thresh:
                    conf_correct += 1

            if conf >= self.conf_thresh:
                conf_total += 1

            # put total at the end, as it's used for a reference for labelset
            total += 1


        print('Correct: ', correct/total, '%\n')
        print('Confidence Accuracy: ', conf_correct/conf_total, '%\n')
        print('Confidence Percentage: ', conf_total/total, '%\n')
        print('Confidently Correct/Total: ', conf_correct/total, '%\n')


class Perceptron:
    def __init__(self, model):
        self.model = model

    def classify(self, features):
        # run model, return 'pos' or 'neg'
        output = self.model.predict(np.array([features,]))[0]
        pos_certainty = output[1] - output[0]

        if pos_certainty >= 0:
            return 'pos'

        return 'neg'

    def test(self, x_test, y_test):
        correct = 0
        total = 0

        for features in x_test:
            pred = self.classify(features)

            if pred == y_test[total]:
                correct += 1

            total += 1

        print('Accuracy: ', correct/total * 100, '%\n')


class RuleCLF:
    # convert this to work with bool tables?
    def __init__(self):
        self.good_words = []
        self.bad_words = []

    def classify(self, features):
        pos_count = 0
        neg_count = 0

        for word, appears in features.items():
            if appears:
                if word in self.good_words:
                    pos_count += 1
                elif word in self.bad_words:
                    neg_count +=1

        certainty = pos_count - neg_count

        if certainty >= 0:
            return 'pos'

        return 'neg'

    def test(self, data_set):

        correct = 0
        total = 0

        feature_set = [a for (a, b) in data_set]
        label_set = [b for (a, b) in data_set]

        for features in feature_set:
            # self.classify(
            pass


# def functions

def test_perceptron(data_set, vocab_count, clf):
    # intake data, and split up
    # make sure features and lables are properly prepared
    # convert data into flat vec data
    # labels are fine?
    features = [a for (a, b) in data_set]
    labels = [b for (a, b) in data_set]

    vec_features = []

    for feature in features:
        vec = np.zeros(vocab_count)
        for datum in feature:
            vec[datum] = 1

        vec_features.append(vec)

    clf.test(vec_features, labels)

    


if __name__ == '__main__':

    # test models
    # load test data and run on RuleCLF

    # load perceptron model, and build into perceptron class
    # test perceptron class

    # load all other models, and pass them along with perceptron to voteCLF
    # test voteclf

    test_data_dir = 'train_data/preped_test_data.pickle'
    lexicon_dir = 'train_data/lexicon.pickle'
    
    test_dataset = pickle.load(open(test_data_dir, 'rb'))
    lexicon = pickle.load(open(lexicon_dir, 'rb'))

    vocab_count = len(lexicon.wordset)

    pc_dir = 'train_data/Perceptron_Model'

    perc_model = keras.models.load_model(pc_dir)
    perc_clf = Perceptron(perc_model)

    # test_perceptron(test_dataset, vocab_count, perc_clf)

    rf_dir = 'saved_models/Random_Forest_Model.pickle'
    bnb_dir = 'saved_models/Bernoulli_Naive_Bayes_Model.pickle'
    mnb_dir = 'saved_models/Multinomial_Naive_Bayes_Model.pickle'
    lr_dir = 'saved_models/Logistic_Regression_Model.pickle'
    sv_dir = 'Support_Vector_Machine_Model.pickle'
    sgd_dir = 'Stoichastic_Gradient_Descent_Model.pickle'

    rf = pickle.load(open(rf_dir, 'rb'))
    bnb = pickle.load(open(bnb_dir, 'rb'))
    mnb = pickle.load(open(mnb_dir, 'rb'))
    lr = pickle.load(open(lr_dir, 'rb'))
    sv = pickle.load(open(sv_dir, 'rb'))
    sgd = pickle.load(open(sgd_dir, 'rb'))

    for i in range(10):
        thresh = (0.1 * i) + 0.1
        print('Confidence Threshhold:\n', thresh)
        vote_clf = VoteClf(bool_clfs=[rf, bnb, mnb, lr, sv, sgd], onehot_clfs=[perc_clf],
                           lexicon=lexicon, conf_thresh=thresh)

        vote_clf.test(test_dataset)

        print('\n===================================================================\n')

    







