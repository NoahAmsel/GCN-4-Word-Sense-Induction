import os
import cPickle
import numpy as np
import json
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

class Induce():
    def __init__(self, graph_prefix):
        self.graph_prefix = graph_prefix
        with open(self.graph_prefix+"_params", 'r') as f:
            params = cPickle.load(f)
            self.__dict__.update(params)

    def node2vec(self, input=None, output=None):
        if input is None:
            input = self.graph_prefix + "_adjlist"
        if output is None:
            output = self.graph_prefix + "_embedding"

        #node2vec_path = "../snap/examples/node2vec/node2vec"
        node2vec_path = "../veles/snap/examples/veles/veles"
        self.input = input
        self.output = output
        self.dims = 128
        self.walklength = 80
        self.walkspersource = 10
        self.contextsize = 4 #make it 5?
        self.p = 1.0     # lower = likely to backtrack
        self.q = 1.0     # lower = explore whole community

        with open(self.graph_prefix+"_node2vec_params", 'wb') as f:
            cPickle.dump(self.__dict__, f, 2)

        comm = ("{bin} -i:{input} -o:{output} -d:{dims} " +
                "-l:{walklength} -r:{walkspersource} " +
                "-k:{contextsize} -p:{p} -q:{q} -w").format(bin=node2vec_path, input=self.input,
                                                          output=self.output, dims=self.dims, walklength=self.walklength,
                                                          walkspersource=self.walkspersource, contextsize=self.contextsize,
                                                          p=self.p, q=self.q)
        print(comm)
        status = os.system(comm)
        self.embs2np(self.output, self.output+".npy")

    def embs2np(self, embeddings_file, npy):
        with open(embeddings_file, "r") as f:
            line = f.readline()
            words, dims = line.split()[:2]
            embeds = np.zeros((int(words),int(dims)))
            line = f.readline()
            while line:
                sp = line.split()
                row = int(sp[0])
                embeds[row,:] = [float(s) for s in sp[1:]]
                line = f.readline()
        np.save(npy, embeds)

    def read_embs(self, npy=None, labels=None):
        if npy is None:
            npy = self.graph_prefix + "_embedding.npy"
        if labels is None:
            labels = self.graph_prefix + "_labels"

        self.embeds = np.load(npy)

        with open(self.graph_prefix+"_labels", 'r') as f:
            self.node2ix = json.load(f)

    def cluster(self, X, cluster_method, scorer, normalize=False, low=2, high=10):
        if cluster_method == "affinity":
            aff = AffinityPropagation(preference=0, affinity='precomputed').fit(cosine_similarity(X))
            bestn = max(aff.labels_)+1
            bestscore = scorer(X, clusterer.labels_)
            print "\t", bestn, bestscore
            labels = aff.labels_
        else:
            centroids = None
            labels = None
            bestn = None
            bestscore = -10000

            if cluster_method == "kmeans":
                clusterer = KMeans(n_init=10, n_jobs=-1)
            elif cluster_method == "ward":
                clusterer = AgglomerativeClustering(linkage='ward')
            elif cluster_method == "cos":
                clusterer = AgglomerativeClustering(linkage='average', affinity='cosine')

            for n in range(low, high+1):
                clusterer.n_clusters = n
                clusterer.fit(X)
                score = scorer(X, clusterer.labels_)
                print "\t", n, score
                if score > bestscore:
                    if cluster_method == "kmeans":
                        centroids = clusterer.cluster_centers_
                    labels = clusterer.labels_
                    bestn = n
                    bestscore = score

        if cluster_method != "kmeans":
            centroids = []
            for i in range(bestn):
                clust = X[labels==i,:]
                if normalize:
                    clust = preprocessing.normalize(clust, axis=1)
                centroids.append(clust.mean(axis=0))
            centroids = np.array(centroids)

        return centroids, bestscore, bestn

    def get_edge_embs(self, word):
        edgenode_ids = []
        for other in self.targets2edges[word]:
            if word <= other:
                edgenode_ids.append(self.node2ix[word+"_"+other])
            else:
                edgenode_ids.append(self.node2ix[other+"_"+word])

        return self.embeds[edgenode_ids]

    def induce_targets(self, cluster_method, scorer):
        '''
        cluster_method should be a string, one of "kmeans", "ward", or preferably "cos"
        scorer should be one of calinski_harabaz_score, or silhouette_score
        the function itself), not the string
        '''
        target2senses = {}
        for t in self.targets:
            centroids, bestscore, bestn = self.cluster(self.get_edge_embs(t), cluster_method, scorer)
            print t, bestn, bestscore
            target2senses[t] = centroids
        self.target2senses = target2senses
        return target2senses


if __name__ == "__main__":
    """
    ind = Induce("dumps/monday_unstemmed")
    ind.read_embs()
    # to get embedding of "word", use this
    ind.embeds[ind.node2ix["word"]]
    # to get a matrix of sense embeddings of "target", use:
    target2senses = ind.induce_targets("kmeans", calinski_harabaz_score)
    target2senses["target"]
    # list of target words:
    ind.targets
    # list of all words
    ind.good_words
    # list of all words and newly minted word/edges
    ind.node2ix.keys()
    """


    ind = Induce("dumps/monday_unstemmed")
    #ind.node2vec()
    ind.read_embs()

    scorers = {"calinski":calinski_harabaz_score, "silhouette":silhouette_score}

    #reload(induce)
    #ind = induce.Induce("dumps/short")
    #ind.read_embs("dumps/short_emb.npy", "dumps/short_labels")
