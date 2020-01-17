#!python3


'''

database class for use with the other scripts

    make sure it is mutable!!

    contains data, which other scripts can access

    each script will contain a reference to a single db isntantiated at the
        beggining


data needed:
    all full tweets
    tweet texts?
    (text, sentiment, date) tuples       # don't need text here?
    confident_tweets

'''

import time
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from collections import Counter

class TweetDataBase:
    def __init__(self, li_in, clf_in, save_dir, keywords, gr_out, save_point=500, count_point=100):
        self.full_tweets = []           # contains all tweets
        self.texts = []                 # contains all texts
        self.sentiment_archive = []     # contains high conf tweets and sentiments

        self.keywords = keywords
        self.common_words_over_time = []          # contains the most commonly used words other than keywords
        self.common_nouns_over_time = []
        self.common_adjs_over_time = []

        self.pos_words_over_time = []
        self.pos_nouns_over_time = []
        self.pos_adjs_over_time = []

        self.neg_words_over_time = []
        self.neg_nouns_over_time = []
        self.neg_adjs_over_time = []

        self.stop_words = []
        self.generate_stop_words()
        
        
        # self.time = time.time()
        self.li_in = li_in
        self.clf_in = clf_in
        self.gr_out = gr_out
        
        self.save_dir = save_dir
        
        self.save_point = save_point
        self.save_count = 0

        self.count_point = count_point
        self.words_count = 0


        self.pos_texts = []
        self.neg_texts = []

    '''
    def add_tweet(self, tweet):
        if len(tweet.keys()) > 1:
            self.full_tweets.append(tweet)
            try:
                self.new_tweets.append(tweet['text'].lower())
            except Exception as e:
                print(e)
                for key in tweet.keys():
                    print(key)
                raise

        if (time.time() - self.time) > 20:
            print(len(self.new_tweets))
            print(len(self.new_sentiment))
            self.time = time.time()
    '''

    def generate_stop_words(self):
        self.stop_words = list(set(stopwords.words('english')))
        
        self.stop_words.append('.')
        self.stop_words.append(':')
        self.stop_words.append('rt')
        self.stop_words.append('?')
        self.stop_words.append('!')
        self.stop_words.append(',')
        self.stop_words.append(';')
        self.stop_words.append("'")
        self.stop_words.append('"')
        self.stop_words.append('/')
        self.stop_words.append(')')
        self.stop_words.append('(')
        self.stop_words.append('@')
        self.stop_words.append('rt')
        self.stop_words.append('’')
        self.stop_words.append('https')
        self.stop_words.append('http')
        self.stop_words.append('#')
        self.stop_words.append('``')
        self.stop_words.append('&')
        self.stop_words.append("n't")
        self.stop_words.append("'s")
        self.stop_words.append('n')
        self.stop_words.append('“')
        self.stop_words.append('...')
        self.stop_words.append(' ')
        self.stop_words.append('\n')
        self.stop_words.append("it's")
        self.stop_words.append("let's")
        self.stop_words.append('')
        self.stop_words.append('-')
        self.stop_words.append('it')
        self.stop_words.append('e')
        self.stop_words.append("it’s")

        for keyword in self.keywords:
            self.stop_words.append(keyword)
            self.stop_words.append(keyword.lower())
            self.stop_words.append(keyword.upper())

            for word in keyword.split(' '):
                self.stop_words.append(word)
                self.stop_words.append(word.lower())
                self.stop_words.append(word.upper())

    def add_data(self, data):
        # intake the full data from twitter listener
        # pull out text data
        # build up a database of tweet texts
        self.full_tweets.append(data)

        if 'extended_tweet' in data:
            text = data['extended_tweet']['full_text'].lower()
        else:
            text = data['text'].lower()

        self.texts.append(text)


    def monitor(self):
        # check listener
        # add any new data to self.full_tweets
        # check clf
        # add any new data
        # if amount of new data > x, save
        while True:
            while self.li_in.poll():
                # self.full_tweets.append(self.li_in.recv())
                self.add_data(self.li_in.recv())
                self.save_count += 1
                self.words_count += 1

            while self.clf_in.poll():
                datum = self.clf_in.recv()
                self.sentiment_archive.append(datum)
                self.save_count += 1
                self.words_count += 1

                if datum[0] == 'pos':
                    self.pos_texts.append(datum[3])
                elif datum[0] == 'neg':
                    self.neg_texts.append(datum[3])

            if self.words_count > self.count_point:
                self.check_words()
                self.words_count = 0

            if self.save_count > self.save_point:
                print('\nsaving')
                pickle.dump(self, open(self.save_dir, 'wb'))
                self.save_count = 0
                print('db saved')

            # throw in a time.sleep to free up some processing?

    def check_words(self):
        # get the most common words
        all_words = []
        pos_words = []
        neg_words = []

        for text in self.texts[-self.count_point:]:
            text = text.lower()
            for word in text.split(' '):
                if word not in self.stop_words:
                    all_words.append(word)

        for text in self.pos_texts[-self.count_point:]:
            text = text.lower()
            for word in text.split(' '):
                if word not in self.stop_words:
                    pos_words.append(word)

        for text in self.neg_texts[-self.count_point:]:
            text = text.lower()
            for word in text.split(' '):
                if word not in self.stop_words:
                    neg_words.append(word)


        # get noun and adjective sets

        # use counter to find most common word in each set
        # print('\n\n')

        top_three = Counter(all_words).most_common(3)
        top_three = [word[0] for word in top_three]
        self.common_words_over_time.append(top_three)
        # print(top_three)

        top_three.append('all')
        self.gr_out.send(top_three)


        top_three = Counter(pos_words).most_common(3)
        top_three = [word[0] for word in top_three]
        self.pos_words_over_time.append(top_three)
        # print(top_three)

        top_three.append('pos')
        self.gr_out.send(top_three)


        top_three = Counter(neg_words).most_common(3)
        top_three = [word[0] for word in top_three]
        self.neg_words_over_time.append(top_three)
        # print(top_three)

        top_three.append('neg')
        self.gr_out.send(top_three)

        # print('\n\n')

        return
        







        
