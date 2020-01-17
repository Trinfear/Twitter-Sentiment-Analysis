#!python3


'''

testing lister stuff
find a way to search by keyword, then limit by location

'''

import datetime

import time
import json
from tweepy import Stream
from Twitter_API_Keys import *  # locational of twitter keys, kept out for security
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


class Listener(StreamListener):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.count = 0
        self.start_time = time.time()
        self.place_count = 0
        self.full_place = 0
        self.coord_count = 0
        self.geo_count = 0
    
    def on_data(self, data):
        all_data = json.loads(data)
        # print(all_data)
##        print(all_data['coordinates'])
##        print(all_data['geo'])
##        print(all_data['place']['full_name'])
##        print('\n')
##        print(all_data['source'])
##        print('\n\n')
##
##        print(all_data['geo'])
##
##        print('\n\n\n')
##
##        for key in all_data.keys():
##            print(key)
##
##        print('\n\nthats all folks')
##        # quit()

##        for key in all_data['user'].keys():
##            print(key)
####
##        return False
    
        self.count += 1

        if 'place' in all_data:
            self.place_count += 1
            if all_data['place'] is not None:
                self.full_place += 1

        if 'user' in all_data:
            if 'location' in all_data['user']:
                if all_data['user']['location'] is not None:
                    self.coord_count += 1

##        if 'geo' in all_data:
##            if all_data['geo'] is not None:
##                self.geo_count += 1
        

        if time.time() - self.start_time > 180:
            print(self.count)
            print(self.full_place)
            print(self.coord_count)
            # print(self.geo_count)
            return False

        return True



if __name__ == '__main__':

    print('here')
    
    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)

##    twitterStream = Stream(auth, Listener())
##
##    start_time = time.time()
##    count = 0
##    # colorado:  36.9990° N, 109.0452° W    N 41° 00.141 W 102° 03.094
##    twitterStream.filter(locations=[-109.0452,36.990,-102.094,41.0141])
##
##    print(count)
##
##    start_time = time.time()
    count = 0
    twitterStream = Stream(auth, Listener(wait_on_rate_limit=True,wait_on_rate_limit_notify=True))
    twitterStream.filter(track=['trump'])

    # print(count)

    print('there')


