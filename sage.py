import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np
import os

#lemmatizer = nltk.stem.WordNetLemmatizer()
#stemmer = nltk.stem.SnowballStemmer("english")

def make_cooc_graph(adjfile, thresh, edge_thresh):
    G = nx.Graph()

    with open(adjfile,"r") as file:
        for line in file:
            u, v, w = line.split(" ")
            # todo implement edge_thresh
            if w > thresh:
                G.add_edge(u, v, weight=w)
    nx.set_node_attributes(G, 'test', False)
    nx.set_node_attributes(G, 'val', False)
    #insurance
    G.node['0']['test'] = True
    G.node['1']['val'] = True
    return G

def make_embedding_mat(labelsfile, glove="glove/glove.6B.50d.txt"):
    with open(labelsfile, "r") as file:
        node2ix = json.load(file)

    ix2vec = np.zeros((len(node2ix), 50))
    with open(glove) as file:
        for line in file:
            toks = line.split(" ")
            if toks[0] in node2ix:
                ix2vec[node2ix[toks[0]],:] = np.array([float(s) for s in toks[1:]])

    return ix2vec

def save_cooc_graph(prefix, thresh, edge_thresh):
    with open(prefix+"-G.json","w") as out:
        G = make_cooc_graph(prefix+"_adjlist", thresh, edge_thresh)
        json.dump(json_graph.node_link_data(G), out)

    with open(prefix+"-id_map.json","w") as out:
        json.dump({str(i):i for i in range(G.number_of_nodes())}, out)

    with open(prefix+"-class_map.json","w") as out:
        #junk
        json.dump({str(i):[0] for i in range(G.number_of_nodes())}, out)

    ix2vec = make_embedding_mat(prefix+"_labels")
    np.save(prefix+"-feats.npy", ix2vec)

def generate_walks(prefix):

    print("Running walks on "+prefix)
    os.system('export PYTHONPATH="${PYTHONPATH}:/data/lily/hll5/finalproject/GraphSAGE"')

    comm = "python ../GraphSAGE/graphsage/utils.py {0}-G.json {0}-walks.txt".format(prefix)
    status = os.system(comm)
    print("Ran walks on "+prefix)
    return status

def runGCN(prefix):
    print("Starting GCN on "+prefix)

    model = "graphsage_mean"
    comm = ("python /data/lily/hll5/finalproject/GraphSAGE/graphsage/unsupervised_train.py " +
            "--train_prefix {0} " +
            "--base_log_dir {1} " +
            "--dim_1 64 " +
            "--model {2} --max_total_steps 1000 ").format(prefix, "sage-"+pref.split("/")[-1], model)
    print(comm)

    status = os.system('export PYTHONPATH="${PYTHONPATH}:/data/lily/hll5/finalproject/GraphSAGE"; '+comm)
    #status = os.system(comm)
    print("Ran GCN on "+prefix)

if __name__ == "__main__":
    """
    This script reads information about the 3 graphs created in cooccurrence.py,
    looks up pretrained Glove embeddings to use as inputs, produces
    GraphSage embeddings, and saves them to disk.
    """

    graphs = ["dumps/monday_unstemmed", 'dumps/tuesday_unstemmed_BOTHhighthresh', "dumps/tuesday_unstemmed_highN"]
    thresh = np.log(6)
    edge_thresh = thresh
    for pref in graphs:
        save_cooc_graph(pref, thresh, edge_thresh)
        generate_walks(pref)
        runGCN(pref)
