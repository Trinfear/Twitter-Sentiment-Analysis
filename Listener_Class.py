#!python3

'''

twitter listener class
    generate simple listener class to be imported elsewhere


todo:
    figure out setting up multiple seperate streams
    figure out the tweet density issue
    find ways to limit tweets more dynamically, ie by area


streaming by area:
    can create a bounded box py providing two coordinate pairs
    start with southwest pair, then northeast pair
    examples:
        San Fransisco: -122.75,36.8,-121.75,37.8

        locations = -122.75,36.8,-121.75,37.8        # make this into a list?


        colorado:  36.9990° N, 109.0452° W    N 41° 00.141 W 102° 03.094

                Latitude	37°N to 41°N
                Longitude	102°02'48"W to 109°02'48"W

'''

import datetime

import time
import json
from tweepy import Stream
from Twitter_API_Keys import *  # locational of twitter keys, kept out for security
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener



class Listener(StreamListener):  # this needs to be here to properly call funcs
    # def an init to give global variables to store data?
    def __init__(self, to_db, to_clf, keywords, **kwargs):
        super().__init__(kwargs)
        self.to_db = to_db
        self.to_clf = to_clf
        # self.count = 0
        self.keywords = keywords
        print('Listener started')

        self.loc_search = False

        self.loc_name = 'colorado'
        self.loc_short = 'co'

    def on_data(self, data):
        # print('still here....')
        #good = self.check_keywords(data.text.lower())
        all_data = json.loads(data)
        good = not self.loc_search  # simpler to put it here then in two else loops

        if self.loc_search:
            if 'place' in all_data or 'user' in all_data:
                if all_data['place'] is not None:
                    good = self.check_locations(all_data['place']['full_name'].lower())
                    
                if 'location' in all_data['user']:
                    if all_data['user']['location'] is not None:
                        good = self.check_locations(all_data['user']['location'].lower())
        
        if good:
            # print(self.count)
            # print('\n')
            
            if len(all_data.keys()) > 1:
                self.to_db.send(all_data)
                # print('\nsent to db')
                # self.count += 1
                # print(self.count)
                
                try:
                    # add a check for if there is 'full text' or something?
                    if 'extended_tweet' in all_data:
                        text = all_data['extended_tweet']['full_text'].lower()
                    else:
                        text = all_data['text'].lower()
                    
                    assert text is not None
                    assert text is not ''
                    # print(text)
                    self.to_clf.send(text)
                except Exception as e:
                    print(e)
                    print('\n\n')
                    for key in all_data.keys():
                        print(key)
                    print('\n\n')
                    
            elif len(all_data.keys()) <= 1:
                print('short data: ')
                print(all_data.keys())
                print('\n')

            # print('returning')
            time.sleep(0.05)  # if issue is process ending before data upload, pause?
            return True
    
    def on_error(self, status):
        print(status)
        return

    def check_keywords(self, text):
        
        for word in self.keywords:
            if word in data.text.lower():
                return True

        return False

    def check_locations(self, name):

        # print(name)
        if name[-2:] == self.loc_short:
            # print('\ngot one\n')
            return True

        if self.loc_name in name:
            # print('\ngot one\n')
            return True

        return False

    def on_exception(self, exception):
        print(exception)
        return

        '''
        try:
            tweet = all_data['text'].lower()
            self.all_tweets.append(tweet)
            self.all_times.append(datetime.datetime.now())
            sentiment, conf = pred_tweet(tweet)
            # move prediction our of here entirely?
            # just have it save text to some database and everything else references db?
        except Exception as e:
            # print('error')
            print(e)
            return False

        if conf >= 0.8:
            self.conf_tweets.append(tweet)
            self.sentiments.append(sentiment)
            self.conf_times.append(datetime.datetime.now())
        '''


'''

add func to auto run listener and restart on errors?

'''

def run_listener():
    while True:
        try:
            # run the listener from here
            pass
        except:
            continue


##
##def pred_tweet(text):
##    # intake tweet
##    # turn into index data
##    # pass to classifiers
##    # return prediction and confidence
##    pass













