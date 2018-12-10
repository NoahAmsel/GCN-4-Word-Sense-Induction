from bs4 import BeautifulSoup
import nltk
import os
import cPickle
from collections import Counter, defaultdict
import itertools
import numpy as np
import json

class Cooccurrence():
    def __init__(self, savepatah, stem=True):
        self.dictionary = set(nltk.corpus.words.words())
        self.stemmer = nltk.stem.SnowballStemmer("english")

        self.savepath = savepath
        self.stem=stem

        self.wordcounts = Counter()
        self.pairs = Counter()
        self.n_sentences = 0
        self.targets = []

    def load(self):
        with open(self.savepath, 'rb') as f:
            tmp_dict = cPickle.load(f)
        self.__dict__.update(tmp_dict)

    def save(self):
        with open(self.savepath, 'wb') as f:
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
        print(self.n_sentences+" processed sentences. Finished.")

    def process_sentence(self, sent):
        # should we encode to ascii? get errors using str(sent) ...
        # LEMMATIZE??
        # STOPWORDS?
        if self.stem:
            word_bag = [self.stemmer.stem(w) for w in nltk.word_tokenize(sent.lower()) if w in self.dictionary]
        else:
            word_bag = [w for w in nltk.word_tokenize(sent.lower()) if w in self.dictionary]
        # remove duplicates. sort b/c key-pairs in all_words must have consistent order
        word_bag = list(set(word_bag))
        word_bag.sort()
        self.wordcounts.update(word_bag)
        self.pairs.update(itertools.combinations(word_bag,2))
        self.n_sentences += 1

    def make_pmi(self, topn=10000, thresh=np.log(2), expand_thresh=None):
        self.topn = topn
        self.thresh = thresh
        if self.expand_thresh is None:
            self.expand_thresh = self.thresh

        good_words, _ = zip(*self.wordcounts.most_common(self.topn))
        self.good_words = set(good_words)
        self.all_nodes = set()
        self.targets2edges = defaultdict(list)

        self.pmi_mat = {}
        lgn = np.log(self.n_sentences)
        for key, val in self.pairs.iteritems():
            if key[0] in self.good_words and key[1] in self.good_words:
                pmi = np.log(val) - np.log(self.wordcounts[key[0]]) - np.log(self.wordcounts[key[1]]) + lgn

                if pmi > self.thresh:
                    self.all_nodes.update(key) #adds both

                    if key[0] in self.targets or key[1] in self.targets and pmi > self.expand_thresh:
                        edgelabel = key[0]+"_"+key[1]
                        self.all_nodes.add(edgelabel)
                        if key[0] in self.targets:
                            self.targets2edges[key[0]].append(key[1])
                        if key[1] in self.targets:
                            self.targets2edges[key[1]].append(key[0])
                        self.pmi_mat[(key[0],edgelabel)] = pmi
                        self.pmi_mat[(edgelabel,key[1])] = pmi

                    # could restore else here:
                    self.pmi_mat[key] = pmi

        self.node2ix = {node: ix for ix, node in enumerate(self.all_nodes)}
        self.save()

    def write_adj_list(self, graph_prefix):
        with open(graph_prefix+"adjlist", "w") as out:
            for key, val in self.pmi_mat.iteritems():
                out.write("{0} {1} {2}\n".format(self.node2ix[key[0]], self.node2ix[key[1]], val))

        with open(graph_prefix+"_labels", "w") as out:
            json.dump(self.node2ix, out)

        with open(graph_file+"_params", 'wb') as out:
            cPickle.dump({"good_words":self.good_words,
                          "targets": self.targets,
                          "targets2edges": self.targets2edges,
                          "n_sentences": self.n_sentences,
                          "topn": self.topn,
                          "thresh": self.thresh,
                          "expand_thresh": self.expand_thresh,
                          "savepath": self.savepath}, out, 2)

if __name__ == "__main__":
    co = Cooccurrence("dumps/monday_stemmed.pkl")
    co.read2010train()
    co.make_pmi(n=20000)
    co.write("dumps/monday_stemmed")
