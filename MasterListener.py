#!python3

'''

master script

    generate perceptron
    generate voteclf

    generate stream listener

    generate graphing

    multithread graphing, listener, and analysis

'''

# imports ============================================================================

##import sys
##print(sys.version)

# other personal scripts
from WordLexicon import *
from CLF_Classes import VoteClf, Perceptron
from Listener_Class import Listener
from Twitter_API_Keys import *  # locational of twitter keys, kept out for security
from DataBase_Class import TweetDataBase as TDB
from Graphing_Class import *
from ClassifierClass import Classifier as CLF_C

# general use scripts
import time
import pickle
import threading
import numpy as np
import datetime as dt
import multiprocessing
from multiprocessing import Pipe, Process
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# task specific scripts
import keras
import tensorflow as tf

from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier as sk
from sklearn.linear_model import LogisticRegression, SGDClassifier


import json
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

# settings imports

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# global variables ===================================================================

# dirs

rf_dir = 'saved_models/Random_Forest_Model.pickle'
bnb_dir = 'saved_models/Bernoulli_Naive_Bayes_Model.pickle'
mnb_dir = 'saved_models/Multinomial_Naive_Bayes_Model.pickle'
lr_dir = 'saved_models/Logistic_Regression_Model.pickle'
sv_dir = 'Support_Vector_Machine_Model.pickle'
sgd_dir = 'Stoichastic_Gradient_Descent_Model.pickle'

pc_dir = 'train_data/Perceptron_Model'

lexicon_dir = 'train_data/lexicon.pickle'

# other

keywords = ['hong kong protestors', 'hong kong riots', 'hk rioters', 'hk protestors', 'hong kong rioters', 'hong kong protests']
# keywords = ['trump', 'impeach', 'impeachment']


# funcs ==============================================================================


def run_listener(keyword_set, auth, a, b, keywords):

    while True:
        try:
            twitterStream = Stream(auth, Listener(a, b, keywords, wait_on_rate_limit=True, wait_on_rate_limit_notify=True))
            twitterStream.filter(track=keyword_set)
        except:
            print('stream error')
            continue

def change_more_settings():
    # use location
    # graph sizes?
    # if certain graphs should be included?
    settings_dict = {}
    return settings_dict


def start_user_input():
    # things to get:
        # keywords
        # location

    new_words = []
    keyword_count = int(input('How many keywords are to be used?\n'))

    for i in range(keyword_count):
        new_words.append(input('Enter a keyword: ').lower())

    while True:
        answ = input('change any other settings? (yes or no)\n').lower()

        if answ == 'yes':
            settings_dict = change_more_settings()
            return new_words, settings_dict
        elif answ == 'no':
            return new_words, None


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # keywords = start_user_input()

    db_save_dir = 'db_saves/db_save_' + str(time.time()) + '.pickle'

    li_to_db, db_from_li = Pipe([True])
    li_to_clf, clf_from_li = Pipe([True])
    clf_to_db, db_from_clf = Pipe([True])
    clf_to_gr, gr_from_clf = Pipe([True])

    db_to_gr, gr_from_db = Pipe([True])

    tweet_db = TDB(db_from_li, db_from_clf, db_save_dir, keywords, db_to_gr)

    grapher = GraphHub(gr_from_clf, gr_from_db, keywords)

    perc_model = keras.models.load_model(pc_dir)
    perc_clf = Perceptron(perc_model)

    lexicon = pickle.load(open(lexicon_dir, 'rb'))

    rf = pickle.load(open(rf_dir, 'rb'))
    bnb = pickle.load(open(bnb_dir, 'rb'))
    mnb = pickle.load(open(mnb_dir, 'rb'))
    lr = pickle.load(open(lr_dir, 'rb'))
    sv = pickle.load(open(sv_dir, 'rb'))
    sgd = pickle.load(open(sgd_dir, 'rb'))

    vote_clf = VoteClf(bool_clfs=[rf, bnb, mnb, lr, sv, sgd], onehot_clfs=[perc_clf],
                           lexicon=lexicon)

    clf = CLF_C(clf_from_li, clf_to_db, clf_to_gr, vote_clf, lexicon)

    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)

    print('\n======================\n', keywords, '\n======================\n')

    # twitterStream = Stream(auth, Listener(li_to_db, li_to_clf))

    
    # multiprocessing.Process(twitterStream.filter(track=keywords, is_async=True)).start()
    multiprocessing.Process(target=run_listener, args=[keywords, auth, li_to_db, li_to_clf, keywords]).start()
    # multiprocessing.Process(target=clf.monitor).start()
    multiprocessing.Process(target=grapher.run_animate).start()
    multiprocessing.Process(target=tweet_db.monitor).start()
    clf.monitor()
##    cd "onedrive\desktop\sentiment analysis\gen 3 scripts"
##    "C:\\Users\tonyt\Python3\python.exe" MasterListener.py

    

    
