#!python3

'''

twitter graphing class 

'''


import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

plt.style.use('seaborn-notebook')


class GraphHub:
    def __init__(self, clf_in, db_in, keywords, calc_window=50, graph_window=200, tick_space=100,
                 use_old=False):
        
        self.clf_in = clf_in
        self.db_in = db_in
        
        self.keywords = keywords

        self.calc_window = calc_window
        self.graph_window = graph_window
        self.tick_space = tick_space
        self.data_window = self.graph_window + self.calc_window

        self.pos_votes = 0
        self.neg_votes = 0
        self.total_votes = 0
        self.current_sentiment = 0.5

        self.use_old = use_old

        self.sentiments = []  # contains sentiment at each time step

        self.all_words = []
        self.pos_words = []
        self.neg_words = []

        self.period_start = 0
        self.tweet_count = 0

        self.period_time = 15

        self.rate_counts = []

        self.times = []  # contains date time at each time step

        self.month_dict = {
            1: 'jan',
            2: 'feb',
            3: 'mar',
            4: 'apr',
            5: 'may',
            6: 'june',
            7: 'july',
            8: 'aug',
            9: 'sep',
            10: 'oct',
            11: 'nov',
            12: 'dec'
            }
    

    def get_data(self):
        new_data = []
        while self.clf_in.poll():
            new_data.append(self.clf_in.recv())

        # print('\n\nnew data size: {} \n\n'.format(len(new_data)))
##        if len(new_data) < 1:
##            time.sleep(10)
##            print('zzzz')
        return new_data

    def get_words(self):
        if not self.db_in.poll():
            return None

        new_words = []
        while self.db_in.poll():
            new_words.append(self.db_in.recv())

        return new_words


    def generate_label(self, full_label):
        # intake full label
        # get month, day, hour, minute
        # generate string
        full_label = full_label.split('-')[1:]
        month = self.month_dict[int(full_label[0])]
        day = full_label[1][:2]

        time = full_label[1][3:]

        hour = time[:2]
        if int(hour) > 12:
            hour = str(int(hour) - 12) + time[2:] + ' pm'
        else:
            hour + str(hour) + time[2:] + ' am'

        return (month + ' ' + day + ',\n' + hour)

    def generate_short_label(self, full_label):
        # intake full label
        # get month, day, hour, minute
        # generate string
        full_label = full_label.split('-')[1:]
        # month = self.month_dict[int(full_label[0])]
        # day = full_label[1][:2]

        time = full_label[1][3:]

        hour = time[:2]
        if int(hour) > 12:
            hour = str(int(hour) - 12) + time[2:] + ' pm'
        else:
            hour + str(hour) + time[2:] + ' am'

        return hour


    def run_animate(self):
        # what will data look like?
        
        print('graphing started')

        fig = plt.figure()
        grid = plt.GridSpec(2, 5, wspace=0.3, hspace=0.7)
        ax1 = fig.add_subplot(grid[0, 2:4])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[1, 1])
        ax4 = fig.add_subplot(grid[1, 2])
        ax5 = fig.add_subplot(grid[1, 3])
        ax6 = fig.add_subplot(grid[0, :2])
        ax7 = fig.add_subplot(grid[0, 4])

        full_title = 'Keywords: '

        for word in self.keywords:
            full_title += str(word)
            full_title += ', '

        full_title = full_title[:-2]
        
        fig.suptitle(full_title)

        # add legends for each of the word categories??

        def animate(i):
            # print('gogogogo')
            data = self.get_data()
            words = self.get_words()
            # if len(data) > 0:
                # print('got one in graphing')
            new_time = time.time() - self.period_start
            
            if new_time > self.period_time:
                new_rate = self.tweet_count / new_time
                self.rate_counts.append(new_rate)
                del(new_rate)
                self.tweet_count = 0
                self.period_start = time.time()

            del(new_time)
            for line in data:

                self.tweet_count += 1

                if self.total_votes >= self.calc_window:   # should this be graph window??
                    # in order to keep a rolling average, auto decrease vote counts
                    # print('length reduction, graphing')
                    self.total_votes -= 1
                    # sentiment = pos/whole, so below reduces the votes
                        # proportional to their share of the total
                    self.pos_votes -= self.current_sentiment
                    self.neg_votes -= (1 - self.current_sentiment)

                self.total_votes += 1
                # print(self.total_votes)

                if line[0] == 'pos':
                    self.pos_votes += 1
                    # print('pos')
                elif line[0] == 'neg':
                    # print('neg')
                    self.neg_votes += 1

                self.times.append(line[1])

                try:
                    self.current_sentiment = self.pos_votes/self.total_votes
                    # print(self.current_sentiment, '    ', self.pos_votes, self.total_votes)
                except ZeroDivisionError:
                    print('zero error')
                    self.current_sentiment=0.5

                self.sentiments.append(self.current_sentiment)
                # print('\n\n')

##                if len(self.sentiments) > self.graph_window:
##                    self.sentiments = self.sentiments[1:]
##                    self.times = self.times[1:]

            if words is not None:
                for word_set in words:
                    if word_set[-1] == 'all':
                        self.all_words = word_set[:-1]
                    elif word_set[-1] == 'pos':
                        self.pos_words = word_set[:-1]
                    elif word_set[-1] == 'neg':
                        self.neg_words = word_set[:-1]

            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax5.clear()
            ax6.clear()
            ax7.clear()

            ax1.set_title('Recent Sentiment', fontsize=18)
            ax2.set_title('Current Opinion', fontsize=18)
            ax6.set_title('All Sentiment', fontsize=18)
            ax7.set_title('Tweets Per Second\nOver Time', fontsize=14)

            # ax1.ylabel('% pos tweets')
            
            try:
                ax1.plot(range(len(self.sentiments[-self.graph_window:])),
                         self.sentiments[-self.graph_window:])

                ax6.plot(range(len(self.sentiments[self.calc_window:])), self.sentiments[self.calc_window:])
                
                ax7.plot(range(len(self.rate_counts[2:])), self.rate_counts[2:])

                if len(self.sentiments) > 0:
                    ax2.pie([self.sentiments[-1], (1 - self.sentiments[-1])],
                            colors=['green', 'r'], shadow=True, autopct = '%1.1f%%',
                            explode=(0.05, 0.05))

                
            except ValueError as e:
                print('graphing error')
                print(len(range(0, self.total_votes)))
                print(len(self.sentiments))
                for sent in self.sentiments:
                    assert type(sent) == float or type(sent) == int
                print('\n\n\n\n')


            if len(self.sentiments) > 0:
                try:
                    labels = [item.get_text() for item in ax1.get_xticklabels()]

                    step_size = int(min(len(self.times), self.graph_window) / (len(labels)))

                    for i in range(len(labels)):
                        try:
                            labels[i] = self.generate_short_label(str(self.times[-self.graph_window:][i * step_size])[4:16])
                        except:
                            labels[i] = self.generate_short_label(str(self.times[-self.graph_window:][-1])[4:16])

                    ax1.set_xticklabels(labels)


                    labels = [item.get_text() for item in ax6.get_xticklabels()]

                    step_size = int(len(self.times) / (len(labels) + 1))

                    for i in range(len(labels)):
                        try:
                            labels[i] = self.generate_label(str(self.times[i * step_size])[4:16])
                        except:
                            labels[i] = self.generate_label(str(self.times[-1])[4:16])

                    ax6.set_xticklabels(labels)


                    labels = [item.get_text() for item in ax7.get_xticklabels()]

                    step_size = int(len(self.times) / (len(labels) + 1))

                    for i in range(len(labels)):
                        try:
                            labels[i] = self.generate_label(str(self.times[i * step_size])[4:16])
                        except:
                            labels[i] = self.generate_label(str(self.times[-1])[4:16])

                    ax7.set_xticklabels(labels)
                    

                except Exception as e:
                    print(e)

                # ax1.set_yticklabels(labels)

            for label in ax1.xaxis.get_ticklabels():
                label.set_rotation(45)

            for label in ax6.xaxis.get_ticklabels():
                label.set_rotation(45)

            for label in ax7.xaxis.get_ticklabels():
                label.set_rotation(45)

            
            if len(self.all_words) > 0:
                text_string = 'most common words, all:\n\n' + self.all_words[0] + '\n' + self.all_words[1] + '\n' + self.all_words[2]

                ax3.text(0.5, 0.5, text_string, horizontalalignment='center', verticalalignment='center')

            if len(self.pos_words) > 0:
                text_string = 'most common words, pos:\n\n' + self.pos_words[0] + '\n' + self.pos_words[1] + '\n' + self.pos_words[2]

                ax4.text(0.5, 0.5, text_string, horizontalalignment='center', verticalalignment='center')

            if len(self.neg_words) > 0:
                text_string = 'most common words, neg:\n\n' + self.neg_words[0] + '\n' + self.neg_words[1] + '\n' + self.neg_words[2]

                ax5.text(0.5, 0.5, text_string, horizontalalignment='center', verticalalignment='center')

            ax3.xaxis.set_visible(False)
            ax3.yaxis.set_visible(False)
            ax3.set_frame_on(False)

            ax4.xaxis.set_visible(False)
            ax4.yaxis.set_visible(False)
            ax4.set_frame_on(False)

            ax5.xaxis.set_visible(False)
            ax5.yaxis.set_visible(False)
            ax5.set_frame_on(False)
                    

        ani = animation.FuncAnimation(fig, animate)
        plt.show()



