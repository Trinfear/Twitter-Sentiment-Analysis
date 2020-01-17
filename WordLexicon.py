#!python3

'''

word lexicon class to be loaded by other scripts

'''


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
