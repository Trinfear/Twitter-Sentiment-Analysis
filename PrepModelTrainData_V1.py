#!python3


'''

twitter data model training prep:
    load in and prep positive and negative datasets


load data
pair sentences with a tag for pos/neg
iterate through sentences and remove stop words, and general cleaning
get wordset
    use all words? max word count?
remove sentences which contain non approved words
convert sentences to word indexes
shuffle sentences
return

todo:
    look at utilizing min appearances count and max word count
    split into train and test data

'''

import os
import nltk
import random
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, movie_reviews


# declare global variables

max_word_count = 15000  # should this exist at all?  Just for early testing?
min_appearances_count = 10  # if word appears less than 5 times, drop it?
max_lost_count = 5  # if a review loses more than this many words, drop it?

train_test_split = 0.95  # this probably isn't too high?

pos_txt_dir = 'positive.txt'
neg_txt_dir = 'negative.txt'
vader_dir = 'Vader_Sentiment.txt'
movie_sent_dir = 'imdb-movie-reviews-dataset/aclImdb'
# need to move into 'train' and 'test' and then inside each 'pos' and 'neg'

train_save_dir = 'Gen 3 Scripts/train_data/preped_train_data.pickle'
test_save_dir = 'Gen 3 Scripts/train_data/preped_test_data.pickle'
word_set_save_dir = 'Gen 3 Scripts/train_data/all_words.pickle'
lex_save_dir = 'Gen 3 Scripts/train_data/lexicon.pickle'



# declare classes

class WordLexicon:

    def __init__(self, wordset):
        self.wordset = wordset
        self.forward = {}
        self.reverse = {}
        self.generate_map()

    def generate_map(self):
        forward_dict = {}
        reverse_dict = {}
        
        i = 0  # 0 is used as padding for NN
        for word in self.wordset:
            forward_dict[word] = i
            reverse_dict[i] = word
            
            i += 1

        self.forward = forward_dict
        self.reverse = reverse_dict



# declare functions

def tokenize_data(text, tag):
    # intake data
    # break into senteneces
    # break into words
    # remove stopwords
    # return

    stop_words = set(stopwords.words('english'))
    # look into removing some names and other proper nouns as well?
    data_set = []

    for doc in text.split('\n'):
        data_set.append([[word.lower() for word in word_tokenize(doc)
                          if word not in stop_words],
                          tag])

    return data_set


def vectorize_sentence(sentence, lexicon):
    # intake lexicon and sentence
    # create a new empty list
    # fill list with index ints to correspond to words
    sent_vec = []
    for word in sentence:
        sent_vec.append(lexicon.forward[word])

    
    return sent_vec


def get_vader_words():
    #
    words = []

    file = open(vader_dir, 'r')
    dataset = file.read()
    file.close()

    lines = dataset.split('\n')
    
    line_set = []
    for line in lines:
        line_set.append(line.split('\t'))
    del(lines)

    for line in line_set:
        if len(line) < 4:
            continue
        
        words.append(line[0])
    
    return words


def get_word_set(data_set):
    # iterate through dataset
    # get list of all words
    # get count of all words
    # sort wordset
    # cut all words with appearances below min_appearances_count

    all_words = []

    for review in data_set:

        for word in review[0]:
            all_words.append(word)

    all_words = nltk.FreqDist(all_words)
    print('pretrimmed word count: ', len(all_words))

    used_words = ['paddingpadpad']  # start with this word in position 0
    # so that padding for rnns works out
    # TODO: instead just used all_words[:x] ??

    for word, count in all_words.items():
        if count > min_appearances_count:
            used_words.append(word)

    if len(used_words) > max_word_count:
        all_words = nltk.FreqDist(used_words)
        used_words = list(all_words.keys())[:max_word_count]

##    for word in get_vader_words():
##        if word not in used_words:
##            used_words.append(word)

    return used_words


def trim_sentence(sentence, word_set):
    # iterate through sentence
    # if word is in approved words, add to new
    
    new_sentence = []
    for word in sentence:
        if word in word_set:
            new_sentence.append(word)

    # add a check for losing too many words??

    if len(new_sentence) == 0 or len(sentence) - len(new_sentence) > max_lost_count:
        return None

    return new_sentence


def trim_words(data_set):
    # iterate through all the reviews in the dataset
    # get a list of all words
    # get a count of all words
    # iterate through again and remove words not appearing enough
    # if a sentence loses too many words, remove the review
    # return the cleaned set of reviews

    word_set = get_word_set(data_set)

    new_data = []

    for datum in data_set:
        sent = datum[0]
        new_sent = trim_sentence(sent, word_set)
        if new_sent:
            new_data.append((new_sent, datum[1]))

    return new_data, word_set


def prepare_nltk_movie_review_data():
    # prepares training data from nltk movie corpus

    return [(movie_reviews.words(fileid), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]


def prepare_imdb_movie_data():
    # load into dir/test/pos
        # get full dir list
        # iterate through directories and intake texts
        # label texts with pos
    # do same for:
        # dir/test/neg
        # dir/train/pos
        # dir/train/neg
        # use different labels for negative reviews
    # run thourgh "tokenize_data()"??
    # conglomerate into single list of texts
    # return list

    pos_texts = []
    neg_texts = []

    full_dir = movie_sent_dir + '/test/pos'

    for datum in os.listdir(full_dir):
        txt_dir = full_dir + '/' + datum
        try:
            pos_texts.append(open(txt_dir).read())
        except:
            pass
            # print(txt_dir)
            # break

    full_dir = movie_sent_dir + '/train/pos'

    for datum in os.listdir(full_dir):
        txt_dir = full_dir + '/' + datum
        try:
            pos_texts.append(open(txt_dir).read())
        except:
            pass
            # print(txt_dir)
            # break

    full_dir = movie_sent_dir + '/test/neg'

    for datum in os.listdir(full_dir):
        txt_dir = full_dir + '/' + datum
        try:
            neg_texts.append(open(txt_dir).read())
        except:
            pass
            # print(txt_dir)
            # break

    full_dir = movie_sent_dir + '/train/neg'

    for datum in os.listdir(full_dir):
        txt_dir = full_dir + '/' + datum
        try:
            neg_texts.append(open(txt_dir).read())
        except:
            pass
            # print(txt_dir)
            # break

    stop_words = set(stopwords.words('english'))

    all_reviews = []

    for review in pos_texts:
        all_reviews.append([[word.lower() for word in word_tokenize(review)
                            if word not in stop_words],
                            'pos'])

    for review in neg_texts:
        all_reviews.append([[word.lower() for word in word_tokenize(review)
                             if word not in stop_words],
                            'neg'])

    return all_reviews


def prepare_text_file_data():
    # prepares training data from pos/neg text files

    pos_text = open(pos_txt_dir, 'r').read()
    neg_text = open(neg_txt_dir, 'r').read()

    pos_text = tokenize_data(pos_text, 'pos')
    neg_text = tokenize_data(neg_text, 'neg')

    all_texts = []

    for text in pos_text:
        all_texts.append(text)

    for text in neg_text:
        all_texts.append(text)

    return all_texts


def prepare_vader_data():
    # prepares training data from vader file
    # just label each word as an individual text?
    # labels above 2 = 'pos'
    # labels below -2 = 'neg'
    # drop the rest?
    file = open(vader_dir, 'r')
    dataset = file.read()
    file.close()

    lines = dataset.split('\n')
    
    line_set = []
    for line in lines:
        line_set.append(line.split('\t'))

    # print('vader initial: ', len(lines))
    del(lines)

    dataset = []
    pos_count = 0
    neg_count = 0

    for line in line_set:
        if len(line) < 4:
            # print(line)
            continue
        if float(line[1]) < -1:
            dataset.append(([line[0]], 'neg'))
            neg_count += 1
        elif float(line[1]) > 1:
            dataset.append(([line[0]], 'pos'))
            pos_count += 1

##    print('pos: ', pos_count, '\nneg: ', neg_count)
##    print('total: ', len(dataset), '\n\n\n')
##
##    for line in dataset[:10]:
##        print(line)
##        print('\n')
    # print('vader final: ', len(line_set))

    return dataset


def save_data(train_data, test_data, word_set, lexicon):

    pickle.dump(train_data, open(train_save_dir, 'wb'))
    pickle.dump(test_data, open(test_save_dir, 'wb'))
    pickle.dump(word_set, open(word_set_save_dir, 'wb'))
    pickle.dump(lexicon, open(lex_save_dir, 'wb'))
##    train_file = open(train_save_dir, 'w')
##    for datum in train_data:
##        train_file.write(str(datum))
##        train_file.write('\n')
##    train_file.close()
##    #print('\nsaved training data set\n')
##
##    test_file = open(test_save_dir, 'w')
##    for datum in test_data:
##        test_file.write(str(datum))
##        test_file.write('\n')
##    test_file.close()
##    #print('saved testing data set\n')
##
##    word_file = open(word_set_save_dir, 'w')
##    for datum in word_set:
##        word_file.write(str(datum))
##        word_file.write('\n')
##    word_file.close()
##    #print('saved word bank\n')
##
##    lex_file = open(lex_save_dir, 'wb')
##    pickle.dump(lexicon, lex_file)
##    lex_file.close()
##    #print('saved lexicon\n')


def prepare_all_data():
    # run every other function that gets data from a specific source
    # conglomerate it all into one big set of training data
    # do any extra cleaning
    # create lexicon
    # create bool tables
    # save it

    # for data from sources:
    # get in form [(word_list, label)]

    stop_words = set(stopwords.words('english'))
    all_reviews = []

    for review in prepare_nltk_movie_review_data():
        all_reviews.append(review)

    # print(all_reviews[-1], '\n')

    for review in prepare_text_file_data():
        all_reviews.append(review)

    # print(all_reviews[-1], '\n')

    for review in prepare_vader_data():
        all_reviews.append(review)

    for review in prepare_imdb_movie_data():
        all_reviews.append(review)

    # print(all_reviews[-4], '\n')

    # print('got all datapoints')
    
    random.shuffle(all_reviews)

    cleaned_reviews = []

    for review in all_reviews:
        text = review[0]
        new_text = []
        if len(review) < 2:
            # print(review)
            continue
        for word in text:
            
            if word not in stop_words:
                new_text.append(word)
        cleaned_reviews.append((new_text, review[1]))

    random.shuffle(cleaned_reviews)

    del(all_reviews)

    # print('cleaned all datapoints')

##    all_words = []
##    # instead call func to get this
##
##    for review in cleaned_reviews:
##
##        for word in review[0]:
##            all_words.append(word)
##
##    all_words = list(set(all_words))

    cleaned_reviews, all_words = trim_words(cleaned_reviews)

    #print('generated word set')
    print('final word count: ', len(all_words))

    copy_words = all_words[1:]

    random.shuffle(copy_words)

    all_words = [all_words[0]]
    for word in copy_words:
        all_words.append(word)
    del(copy_words)

    lex = WordLexicon(all_words)

    vec_data = []

    for sent in cleaned_reviews:
        vec_data.append([vectorize_sentence(sent[0], lex), sent[1]])

    train_split = int(train_test_split * len(vec_data))

    train_data = vec_data[:train_split]
    test_data = vec_data[train_split:]
    # print('split data\n')

    save_data(train_data, test_data, all_words, lex)

    return train_data, test_data, all_words, lex


def create_bool_tables(dataset, lexicon):  # should this be in prep data?
    
    # intake a sentence and convert to a onehot bool table akin to format in old script
    # itterate through sentences and generate onehot bool table

    bool_tables = []
    for datum in dataset:
        bool_table = {}
        for word in lexicon.wordset:
            bool_table[word] = (lexicon.forward[word] in datum[0])

        bool_tables.append([bool_table, datum[1]])

    return bool_tables


if __name__ == '__main__':

    # practice running to get data
    # normally this script will be called elsewhere, so this is just for testing

    
    prepare_all_data()
    # train_dataset, test_dataset, wordset, lexicon = prepare_train_data()

    # bool_set = create_bool_tables(train_dataset, lexicon)







