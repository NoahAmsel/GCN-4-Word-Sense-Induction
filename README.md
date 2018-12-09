## making the co-occurrence graph

## node2vec
for weighted graph, just make each line    a, b, weight

## datasets

used in allen paper
https://www.cs.york.ac.uk/semeval-2013/task13/

used in russian paper (https://arxiv.org/pdf/1804.10686.pdf)
https://www.cs.york.ac.uk/semeval2010_WSI/taskdescription.html
The original SemEval 2010 Task 14 used the V-Measure ex- ternal clustering measure (Manandhar et al., 2010). How- ever, this measure is maximized by clustering each sentence into his own distinct cluster, i.e., a ‘dummy’ singleton base- line. This is achieved by the system deciding that every am- biguous word in every sentence corresponds to a different word sense. To cope with this issue, we follow a similar study (Lopukhin et al., 2017) and use instead of the ad- justed Rand index (ARI) proposed by Hubert and Arabie (1985) as an evaluation measure.


Umass also uses
Word Context Relevance (WCR)
Turk bootstrap Word Sense Inventory (TWSI) but cheat by making all senses occur equal number of times

Evaluation metrics
Jaccard Index, Tau and WNDCG
Fuzzy NMI and Fuzzy B-Cubed


once you have a cluster, average to get an embedding


hannah's netid: hll5
