#!python3

'''

gen 3 training script
    basically the same as the other one, but for one hot encoded data

'''

import time
import pickle
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Embedding, Lambda, Dropout


import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# define global variables

just_save_em = False

train_data_dir = 'train_data/preped_train_data.pickle'
test_data_dir = 'train_data/preped_test_data.pickle'
wordset_dir = 'train_data/all_words.pickle'
# lexicon_dir = 'train_data/lexicon.pickle'

pc_dir = 'train_data/Perceptron_Model'


class Perceptron:
    def __init__(self, x_train, y_train):
        self.model = self.generate_model(x_train.shape[1:])
        self.train_model(x_train, y_train)

    def generate_model(self, shape):
        model = Sequential()
        
        model.add(Dense(128, activation='tanh', input_shape=shape))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])

        return model

    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train, verbose=0)

    def test(self, x_test, y_test):
        val_loss, val_acc = self.model.evaluate(x_test, y_test, verbose=0)
        return val_loss, val_acc

    def classify(self, features):
        output = self.model.predict(np.array([features,]))[0]
        pos_certainty = output[1] - output[0]

        # TODO add in more certainty requirements
        if pos_certainty >= 0:
            return 'pos'

        return 'neg'


def load_data():
    train_dataset = pickle.load(open(train_data_dir, 'rb'))
    test_dataset = pickle.load(open(test_data_dir, 'rb'))
    wordset = pickle.load(open(wordset_dir, 'rb'))
    # lexicon = pickle.load(open(lexicon_dir, 'rb'))

    vocab_count = len(wordset)

    x_train = [datum[0] for datum in train_dataset]
    x_test = [datum[0] for datum in test_dataset]

    y_train = [datum[1] for datum in train_dataset]
    y_test = [datum[1] for datum in test_dataset]

    y_train = np.array([(0,1) if datum == 'pos' else (1,0) for datum in y_train])
    y_test = np.array([(0,1) if datum == 'pos' else (1,0) for datum in y_test])

    def collapse_data(data):
        # intake the ordered vector
        # turn into a bool table, basically?
        flat_data = np.zeros(vocab_count)
        for datum in data:
            flat_data[datum] = 1
        
        flat_data[0] = 0

        return flat_data

    x_train = np.array([collapse_data(x) for x in x_train])
    x_test = np.array([collapse_data(x) for x in x_test])

    return x_train, y_train, x_test, y_test


def save(model, acc):
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
            model.save(pc_dir)
            print('saved\n\n')
            return
        elif answ == 'no':
            print('\n\n')
            return
        
        print('I didnt understand...please answer yes or no')


def generate_models(x_train, y_train, x_test, y_test):

    '''
    start_time = time.time()
    
    pc = Perceptron(x_train, y_train)
    pc_acc = pc.test(x_test, y_test)[1]
    print('Perceptron: ', pc_acc)
    
    time_elapsed = time.time() - start_time
    print('time taken: ', time_elapsed, '\n')

    pickle.dump(pc, open(pc_dir, 'wb'))
    print('saved\n\n')
    del(pc)
    del(pc_acc)
    '''

    start_time = time.time()

    model = Sequential()
        
    model.add(Dense(128, activation='tanh', input_shape=x_train.shape[1:]))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])

    model.fit(x_train, y_train, verbose=0)

    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)

    print('Perceptron: ', val_acc)

    time_elapsed = time.time() - start_time
    print('time taken: ', time_elapsed, '\n')

    save(model, val_acc)
    del(model)
    del(val_acc)
    del(val_loss)


if __name__ == '__main__':
    x_tr, y_tr, x_t, y_t = load_data()
    generate_models(x_tr, y_tr, x_t, y_t)





