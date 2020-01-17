#!python3


'''

classifier class

'''

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import time
import datetime as dt


class Classifier:
    def __init__(self, li_in, to_db, to_gr, clf, lexicon):
        self.li_in = li_in
        self.to_db = to_db
        self.to_gr = to_gr
        self.clf = clf
        self.lexicon = lexicon

    def clean_tweet(self, text):
        text = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        
        index_data = []

        lost_count = 0

        for word in text:
            if word in stop_words:
                continue
            if word in self.lexicon.wordset:
                index_data.append(self.lexicon.forward[word])
            else:
                lost_count += 1

        return index_data

    def pred_tweet(self, text):
        text = self.clean_tweet(text)
        sentiment, confidence, votes = self.clf.classify(text)

        return sentiment, confidence

    def monitor(self):
        while True:
            while self.li_in.poll():
                # print('running monitor')
                tweet = self.li_in.recv()
                try:
                    sentiment, confidence = self.pred_tweet(tweet)
                    
                except Exception as e:
                    print('clf error:')
                    print(e)
                    print('\n')
                    continue
            
                post = (sentiment, dt.datetime.now(), confidence, tweet)
                self.to_db.send(post)
                
                if confidence >= 0.5:
                    self.to_gr.send(post)

            # self.li_in.send('ping')
            # print('nothing new here...')
            time.sleep(2)
