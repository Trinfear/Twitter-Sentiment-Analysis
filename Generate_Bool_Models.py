#!python3

'''

gen 3 training script
    each model trained seperately in it's own script

load in data from a save file
train model
test model
save model

TODO:
    change so funcs take the global variables as arguements
    ie make the load data take data dirs as arguments

'''

# imports

from WordLexicon import *

import nltk
import time
import pickle

from multiprocessing import Pool

from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier as sk
from sklearn.linear_model import LogisticRegression, SGDClassifier

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# define global variables

train_data_dir = 'train_data/preped_train_data.pickle'
test_data_dir = 'train_data/preped_test_data.pickle'
lexicon_dir = 'train_data/lexicon.pickle'

just_save_em = False

prev_accs_dir = 'train_data/current_best_accuracies.txt'
# at somepoint start saving such data here, for reference when saving

rf_dir = 'saved_models/Random_Forest_Model.pickle'
bnb_dir = 'saved_models/Bernoulli_Naive_Bayes_Model.pickle'
mnb_dir = 'saved_models/Multinomial_Naive_Bayes_Model.pickle'
lr_dir = 'saved_models/Logistic_Regression_Model.pickle'
sv_dir = 'Support_Vector_Machine_Model.pickle'
sgd_dir = 'Stoichastic_Gradient_Descent_Model.pickle'


def load_data():
    # move this to data prep, and have it save bool table and non bool table data?
    train_dataset = pickle.load(open(train_data_dir, 'rb'))
    test_dataset = pickle.load(open(test_data_dir, 'rb'))
    lexicon = pickle.load(open(lexicon_dir, 'rb'))

    def create_bool_tables(dataset):  # should this be in prep data?
        # intake a sentence and convert to a onehot bool table akin to format in old script
        # itterate through sentences and generate onehot bool table        bool_tables = convert_pool.map(create_table, [datum for datum in dataset])
        
        bool_tables = []
        for datum in dataset:
            bool_table = {}
            for word in lexicon.wordset:
                bool_table[word] = (lexicon.forward[word] in datum[0])

            bool_tables.append([bool_table, datum[1]])

        return bool_tables


    train_data = create_bool_tables(train_dataset)
    del(train_dataset)
    
    test_data = create_bool_tables(test_dataset)
    del(test_dataset)

    print('preped data')

    return train_data, test_data

def save(model, dir_name, acc):
    # get user input, on whether or not to save
        # user presumably decides by comparing accuracy to current best
        # therefore put accuracy as part of save name?
    #
    while True:

        if just_save_em:
            answ = 'yes'
        else:
            answ = input('Save model? Accuracy = {} ("yes" or "no")\n  '.format(acc)).lower()
        if answ == 'yes':
            pickle.dump(model, open(dir_name, 'wb'))
            print('saved\n\n')
            return
        elif answ == 'no':
            print('\n\n')
            return
        
        print('I didnt understand...please answer yes or no')


def generate_models(train_data, test_data):

    start_time = time.time()

    rf = sk(RandomForestClassifier())

    rf.train(train_data)
    rf_acc = nltk.classify.accuracy(rf, test_data)
    print('Random Forest: ', rf_acc)
    
    time_elapsed = time.time() - start_time
    print('time taken: ', time_elapsed, '\n')

    save(rf, rf_dir, rf_acc)
    del(rf)
    del(rf_acc)

    
    start_time = time.time()

    bnb = sk(BernoulliNB())

    bnb.train(train_data)
    bnb_acc = nltk.classify.accuracy(bnb, test_data)
    print('Bernoulli NB: ', bnb_acc)

    time_elapsed = time.time() - start_time
    print('time taken: ', time_elapsed, '\n')

    save(bnb, bnb_dir, bnb_acc)
    del(bnb)
    del(bnb_acc)


    start_time = time.time()

    mnb = sk(MultinomialNB())

    mnb.train(train_data)
    mnb_acc = nltk.classify.accuracy(mnb, test_data)
    print('Multinomial NB: ', mnb_acc)

    time_elapsed = time.time() - start_time
    print('time taken: ', time_elapsed, '\n')

    save(mnb, mnb_dir, mnb_acc)
    del(mnb)
    del(mnb_acc)


    start_time = time.time()

    lr = sk(LogisticRegression())

    lr.train(train_data)
    lr_acc = nltk.classify.accuracy(lr, test_data)
    print('Logistic Regression: ', lr_acc)

    time_elapsed = time.time() - start_time
    print('time taken: ', time_elapsed, '\n')

    save(lr, lr_dir, lr_acc)
    del(lr)
    del(lr_acc)


    start_time = time.time()

    sv = sk(NuSVC())

    sv.train(train_data)
    sv_acc = nltk.classify.accuracy(sv, test_data)
    print('Support Vector Machine: ', sv_acc)

    time_elapsed = time.time() - start_time
    print('time taken: ', time_elapsed, '\n')

    save(sv, sv_dir, sv_acc)
    del(sv)
    del(sv_acc)


    start_time = time.time()

    sgd = sk(SGDClassifier())

    sgd.train(train_data)
    sgd_acc = nltk.classify.accuracy(sgd, test_data)
    print('Stoichastic Gradient Descent: ', sgd_acc)

    time_elapsed = time.time() - start_time
    print('time taken: ', time_elapsed, '\n')

    save(sgd, sgd_dir, sgd_acc)
    del(sgd)
    del(sgd_acc)



if __name__ == '__main__':
    train, test = load_data()
    generate_models(train, test)









    
