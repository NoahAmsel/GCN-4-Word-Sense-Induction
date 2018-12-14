import os
import cPickle
import numpy as np
import json
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

class Induce():
    def __init__(self, graph_prefix):
        self.graph_prefix = graph_prefix
        with open(self.graph_prefix+"_params", 'r') as f:
            params = cPickle.load(f)
            self.__dict__.update(params)

    def node2vec(self, dims, input=None, output=None):
        if input is None:
            input = self.graph_prefix + "_adjlist"
        if output is None:
            output = self.graph_prefix + "_embedding"

        #node2vec_path = "../snap/examples/node2vec/node2vec"
        node2vec_path = "../veles/snap/examples/veles/veles"
        self.input = input
        self.output = output
        self.dims = dims
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

        with open(labels, 'r') as f:
            self.node2ix = json.load(f)

    def read_embs_sage(self, folder, labels=None):
        if labels is None:
            labels = self.graph_prefix + "_labels"

        unordered = np.load(folder+"val.npy")
        order = []
        with open(folder+"val.txt") as f:
            for line in f:
                order.append(int(line))
        self.embeds = unordered[order,:]

        with open(labels, 'r') as f:
            self.node2ix = json.load(f)

    def cluster(self, X, cluster_method, scorer, normalize=False, low=2, high=10, origin=None):
        if origin is not None:
            X -= origin

        if normalize:
            X = preprocessing.normalize(X, axis=1)

        if cluster_method == "affinity":
            aff = AffinityPropagation(preference=0, affinity='precomputed').fit(cosine_similarity(X))
            bestn = max(aff.labels_)+1
            if bestn > 10 or bestn < 2:
                print "\t\t\t ================= OH SHIT ================="
                return self.cluster(X, "cos", scorer, normalize, low, high)
            # \/ this has to go after the if b/c it breaks when there's one label :/
            bestscore = scorer(X, aff.labels_)
            print "\t", bestn, bestscore
            labels = aff.labels_
            centroids = X[aff.cluster_centers_indices_,:]

        elif cluster_method == "dbscan":
            db = DBSCAN(eps=0.2, min_samples=0.1*X.shape[0], metric="cosine", n_jobs=1).fit(X)
            bestn = max(db.labels_)+1
            bestscore = sum(db.labels_==-1)
            print "\t", bestn, bestscore
            labels = db.labels_
            centroids = db.components_

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

        print(np.unique(labels, return_counts=True)[1])

        if centroids is None:
            centroids = []
            for i in range(bestn):
                centroids.append(X[labels==i,:].mean(axis=0))
            centroids = np.array(centroids)
            if normalize:
                centroids = preprocessing.normalize(centroids, axis=1)

        if origin is None:
            return centroids, bestscore, bestn, labels
        else:
            return centroids+origin, bestscore, bestn, labels

    def get_edge_embs(self, word):
        edgenode_ids = []
        for other in self.targets2edges[word]:
            if word <= other:
                edgenode_ids.append(self.node2ix[word+"_"+other])
            else:
                edgenode_ids.append(self.node2ix[other+"_"+word])

        return self.embeds[edgenode_ids]

    def induce_targets(self, cluster_method, scorer, suff="", remove_target=False):
        '''
        cluster_method should be a string, one of "kmeans", "ward", or preferably "cos"
        scorer should be one of calinski_harabaz_score, or silhouette_score
        the function itself), not the string
        '''
        target2senses = {}
        for t in self.targets:
            X = self.get_edge_embs(t)
            if remove_target:
                #t_emb = self.embeds[self.node2ix[t],:]
                t_emb = X.mean(axis=0)
            else:
                t_emb = None
            centroids_and_meta = self.cluster(X, cluster_method, scorer, origin=t_emb)
            print t, centroids_and_meta[1], centroids_and_meta[2]
            target2senses[t] = centroids_and_meta
        self.target2senses = target2senses

        with open(self.graph_prefix+"_WSDdata_"+suff+".pkl", 'wb') as f:
            cPickle.dump({"targets": self.targets,
                          "target2senses": self.target2senses,
                          "allwords": self.good_words,
                          "embeds": self.embeds,
                          "node2ix": self.node2ix}, f, 2)

    def open_induced_pkl(self, pkl):
        with open(pkl, 'r') as f:
            tmp_dict = cPickle.load(f)
        self.__dict__.update(tmp_dict)

    def show_clusters(self):
        num_senses = []
        max_perc = []
        for t in self.targets:
            centroids, bestscore, bestn, labels = self.target2senses[t]
            cluster_sizes = np.unique(labels, return_counts=True)[1]
            num_senses.append(len(cluster_sizes))
            max_perc.append(max(cluster_sizes)/float(sum(cluster_sizes)))
            #print t, np.unique(labels, return_counts=True)[1]

        print "\t", "mean number of senses", np.mean(num_senses)
        print "\t", "average size of biggest cluster (%)", np.mean(max_perc)

def count_clusters():
    files = ["tuesday_unstemmed_highN_WSDdata_64affinity.pkl",
            "tuesday_unstemmed_highN_WSDdata_64cos.pkl",
            "tuesday_unstemmed_highN_WSDdata_64kmeans.pkl",
            "tuesday_unstemmed_highN_WSDdata_64ward.pkl"]
    ind = Induce("dumps/tuesday_unstemmed_highN")
    for pkl in files:
        ind.open_induced_pkl("dumps/"+pkl)
        print(pkl)
        ind.show_clusters()

if __name__ == "__main__":
    """
    This script reads files containing information about a graph, produces
    node2vec embeddings (and reads in GraphSAGE embeddings from another file)
    and performs the induction step using a few different clustering algorithms.
    Then it writes information about the induced senses and sense embeddings to disk
    """

    # do the node 2 vec
    graphs = ["dumps/monday_unstemmed", 'dumps/tuesday_unstemmed_BOTHhighthresh', "dumps/tuesday_unstemmed_highN"]
    dims_list = [64, 128]
    clusterings = ["kmeans", "ward", "cos", "affinity", "dbscan"]
    for g in graphs:
        ind = Induce(g)
        for dims in dims_list:
            ind.node2vec(dims=dims)
            ind.read_embs()
            for clust in clusterings:
                ind.induce_targets(clust, calinski_harabaz_score, str(dims)+clust, remove_target=(clusterings=="cos"))


    # do the graphsage (this requires running the sage.py first)
    graphs = ["dumps/monday_unstemmed", 'dumps/tuesday_unstemmed_BOTHhighthresh', "dumps/tuesday_unstemmed_highN"]
    embs_folder = ["sage-monday_unstemmed/unsup-dumps/graphsage_mean_small_0.000010/",
                    "sage-tuesday_unstemmed_BOTHhighthresh/unsup-dumps/graphsage_mean_small_0.000010/",
                    "sage-tuesday_unstemmed_highN/unsup-dumps/graphsage_mean_small_0.000010/"]
    clusterings = ["kmeans", "ward", "cos", "affinity", "dbscan"]
    for i in range(len(graphs)):
        ind = Induce(graphs[i])
        ind.read_embs_sage(embs_folder[i])
        for clust in clusterings:
            ind.induce_targets(clust, calinski_harabaz_score, "sage_UNCENTERED_"+clust)
