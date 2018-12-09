from bs4 import BeautifulSoup
import nltk
import os
from collections import Counter
import itertools
import numpy as np

class Cooccurrence():
    def __init__(self):
        self.dictionary = set(nltk.corpus.words.words())
        self.wordcounts = Counter()
        self.pairs = Counter()
        self.n_sentences = 0
        self.target_ratio = 1

    def read2010file(self, path):
        with open(path) as fp:
            soup = BeautifulSoup(fp, "html5lib")
        for sentence_tag in soup.body.contents[0].contents:
            self.process_sentence(sentence_tag.text)

    def process_sentence(self, sent):
        # should we encode to ascii? get errors using str(sent) ...
        # LEMMATIZE??
        word_bag = [w for w in nltk.word_tokenize(sent.lower()) if w in self.dictionary]
        # remove duplicates. sort b/c key-pairs in all_words must have consistent order
        word_bag = list(set(word_bag))
        word_bag.sort()
        self.wordcounts.update(word_bag)
        self.pairs.update(itertools.combinations(word_bag,2))
        self.n_sentences += 1

    def pmi(self, n=10000, thresh=np.log(2), expand=[]):
        good_words, _ = zip(*self.wordcounts.most_common(n))
        self.good_words = set(good_words)
        self.word2ix = {k: v for v, k in enumerate(good_words)}

        self.pmi = {}
        lgn = np.log(self.n_sentences)
        for key, val in self.pairs.iteritems():
            if key[0] in self.good_words and key[1] in self.good_words:
                #reverse_key = (keys[1], keys[0])
                both_count = val #+ self.pairs[reverse_key]
                pmi = np.log(both_count) - np.log(self.wordcounts[key[0]]) - np.log(self.wordcounts[key[1]]) + lgn
                #self.pmi[reverse_key] = self.pmi[key]

                if pmi > thresh:
                    if key[0] in expand or key[1] in expand:
                        edgelabel = key[0]+"_"+key[1]
                        self.pmi[(key[0],edgelabel)] = pmi
                        self.pmi[(edgelabel,key[1])] = pmi
                    else:
                        self.pmi[key] = pmi

    def write(self, graph_file):
        with open(graph_file, "w") as out:
            for key, val in self.pmi.iteritems():
                out.write("{0} {1} {2}\n".format(self.word2ix[key[0]], self.word2ix[key[1]], val))

        with open(graph_file+"_labels", "w") as out:
            for ix, word in enumerate(self.good_words):
                out.write("{0} {1}\n".format(ix, word))

if __name__ == "__main__":
    pass
