# %pylab
"""
This is a collection of code snippets used to visualize and analyze the sense embeddings
we produced. (Finding nearest neighbors of words or representative words for a sense,
etc.) It should not be run.
"""








from sklearn.neighbors import NearestNeighbors

# use this with a pre-existing ind object !!
ix2node = {ix:node for node, ix in ind.node2ix.iteritems()}
real_dub_ixs = [ind.node2ix[x] for x in ind.node2ix if x in ind.good_words]
nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine').fit(ind.embeds[real_dub_ixs,:])
def realword_neighbors(w, nbrs_obj):
    global ind, real_dub_ixs, ix2node
    dist, ixs = nbrs_obj.kneighbors(np.array(ind.embeds[ind.node2ix[w],:]).reshape(1,-1))
    for i in range(1,dist.shape[1]): #start at 1 b/c closest neighbor is self
        print ix2node[real_dub_ixs[ixs[0,i]]], dist[0,i]



'''
from sklearn.decomposition import PCA

reload(induce)
ind = induce.Induce("dumps/short")
ind.read_embs("dumps/short_emb.npy", "dumps/short_labels")

data = ind.embeds
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(n_clusters=4, n_init=10).fit(data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_)
'''
#############
tar = "lie"
X = ind.get_edge_embs(tar)
x = ind.embeds[ind.node2ix[tar],:]
X = np.vstack([X,x])
centroids, bestscore, bestn, labels = ind.target2senses[tar]
X = np.vstack([X,centroids])
np.savetxt("visualization_dmps/"+tar+".tsv", X, delimiter="\t")
with open("visualization_dmps/"+tar+"_labels.tsv","w") as f:
    f.write("label\tsense#\n")
    for i in range(len(ind.targets2edges[tar])):
        w = ind.targets2edges[tar][i]
        lab = labels[i]
        f.write(w+"\t"+str(lab)+"\n")
    f.write(tar+"\t"+str(-1)+"\n")
    for sen in range(bestn):
        f.write("SENSE_"+str(sen)+"\t"+str(sen)+"\n")

#############

from sklearn.neighbors import NearestNeighbors

reload(induce)
ind = induce.Induce("dumps/tuesday_unstemmed_highN")
ind.open_induced_pkl("dumps/tuesday_unstemmed_highN_WSDdata_sage_kmeans.pkl")
ind.show_clusters()

ix2node = {ix:node for node, ix in ind.node2ix.iteritems()}
real_dub_ixs = [ind.node2ix[x] for x in ind.node2ix if x in ind.good_words]
def clusters(tar, class_ix, metric):
    centroids, bestscore, bestn, labels = ind.target2senses[tar]
    print tar + " has " + str(bestn) + " senses. Here's number " + str(class_ix)
    X = ind.get_edge_embs(tar)
    words_in_class = np.array(ind.targets2edges[tar])[labels==class_ix]
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto', metric=metric).fit(X[labels==class_ix,:])
    vec = centroids[class_ix, :].reshape(1, -1)
    dist, ixs = nbrs.kneighbors(vec)
    for i in ixs[0]:
        print words_in_class[i]
