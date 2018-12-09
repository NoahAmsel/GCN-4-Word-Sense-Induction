from bs4 import BeautifulSoup
import nltk
import os
import cPickle
from collections import Counter, defaultdict
import itertools
import numpy as np

class Cooccurrence():
    def __init__(self):
        self.dictionary = set(nltk.corpus.words.words())
        self.wordcounts = Counter()
        self.pairs = Counter()
        self.n_sentences = 0
        self.target_ratio = 1
        self.filename = "dumps/cooccurrence.pkl"

    def load(self):
        with open(self.filename, 'rb') as f:
            tmp_dict = cPickle.load(f)
        self.__dict__.update(tmp_dict)

    def save(self):
        with open(self.filename, 'wb') as f:
            cPickle.dump(self.__dict__, f, 2)

    def read2010train(self):
        self.targets = []
        for folder in ["SemEval-2010/training_data/nouns/", "SemEval-2010/training_data/verbs/"]:
            for file in os.listdir(folder):
                self.read2010file(folder+file)
                self.targets.append(file.split(".")[0])
                print(file)
            self.save()


    def read2010file(self, path):
        with open(path) as fp:
            soup = BeautifulSoup(fp, "html5lib")
        for sentence_tag in soup.body.contents[0].contents:
            self.process_sentence(sentence_tag.text)

    def process_sentence(self, sent):
        # should we encode to ascii? get errors using str(sent) ...
        # LEMMATIZE??
        # STOPWORDS?
        word_bag = [w for w in nltk.word_tokenize(sent.lower()) if w in self.dictionary]
        # remove duplicates. sort b/c key-pairs in all_words must have consistent order
        word_bag = list(set(word_bag))
        word_bag.sort()
        self.wordcounts.update(word_bag)
        self.pairs.update(itertools.combinations(word_bag,2))
        self.n_sentences += 1

    def make_pmi(self, n=10000, thresh=np.log(2), expand=[]):
        good_words, _ = zip(*self.wordcounts.most_common(n))
        self.good_words = set(good_words)
        self.new_words = []
        self.targets2edges = defaultdict(list)

        self.pmi_mat = {}
        lgn = np.log(self.n_sentences)
        for key, val in self.pairs.iteritems():
            if key[0] in self.good_words and key[1] in self.good_words:
                #reverse_key = (keys[1], keys[0])
                both_count = val #+ self.pairs[reverse_key]
                pmi = np.log(both_count) - np.log(self.wordcounts[key[0]]) - np.log(self.wordcounts[key[1]]) + lgn
                #self.pmi_mat[reverse_key] = self.pmi_mat[key]

                if pmi > thresh:
                    if key[0] in expand or key[1] in expand:
                        edgelabel = key[0]+"_"+key[1]
                        self.new_words.append(edgelabel)
                        if key[0] in expand:
                            self.targets2edges[key[0]].append(key[1])
                        if key[1] in expand:
                            self.targets2edges[key[1]].append(key[0])
                        self.pmi_mat[(key[0],edgelabel)] = pmi
                        self.pmi_mat[(edgelabel,key[1])] = pmi
                    else:
                        self.pmi_mat[key] = pmi

        self.word2ix = {k: v for v, k in enumerate(good_words+self.new_words)}
        self.save()

    def write(self, graph_file):
        with open(graph_file, "w") as out:
            for key, val in self.pmi_mat.iteritems():
                out.write("{0} {1} {2}\n".format(self.word2ix[key[0]], self.word2ix[key[1]], val))

        with open(graph_file+"_labels", "w") as out:
            for ix, word in enumerate(self.good_words):
                out.write("{0} {1}\n".format(ix, word))

if __name__ == "__main__":
    co = Cooccurrence()
    co.read2010train()
    co.make_pmi(n=20000, expand=co.targets)
    co.write("dumps/sun1am")
