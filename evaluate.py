import os
import numpy as np
import pretrained
import glob
import cPickle
import json
import cooccurrence as co
from scipy import spatial
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from bs4 import BeautifulSoup
import nltk
import time
import cooccurrence

# Use sense embeddings to disambiguate testing data
# Run this file from GCN-4... directory, where it is located
# Modify "all_cluster_files" to run on different sense embeddings
# The name of the file with tags is, by default, "output_(sensefilename)" where sensefilename is 
# the filename found in all_cluster_files

def process_sentence_e(stem,sent):
	dictionary = set(nltk.corpus.words.words())
	stemmer=nltk.stem.SnowballStemmer("english")
	if stem:
		word_bag = [stemmer.stem(w) for w in nltk.word_tokenize(sent.lower()) if w in dictionary]
	else:
		word_bag = [w for w in nltk.word_tokenize(sent.lower()) if w in dictionary]
    # remove duplicates. sort b/c key-pairs in all_words must have consistent order
	word_bag = list(set(word_bag))
	word_bag.sort()
	return word_bag

def closest_sense(context_embedding,senses,metric):
	if metric=="cos":
		curr=spatial.distance.cosine(context_embedding,senses[0,:])
		# print('0 '+str(curr))
	currind=0
	for i in range(1,np.size(senses,0)):
		if metric=="cos":
			dist=spatial.distance.cosine(context_embedding,senses[i,:])
			# print(str(i)+' '+str(dist))
		if dist<curr:
			curr=dist
			currind=i
	return currind

all_cluster_files=["monday_unstemmed_WSDdata_64affinity.pkl",
"monday_unstemmed_WSDdata_64cos.pkl",
"monday_unstemmed_WSDdata_64kmeans.pkl",
"monday_unstemmed_WSDdata_64ward.pkl",

"tuesday_unstemmed_BOTHhighthresh_WSDdata_64affinity.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_64cos.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_64kmeans.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_64ward.pkl",

"tuesday_unstemmed_highN_WSDdata_64affinity.pkl",
"tuesday_unstemmed_highN_WSDdata_64cos.pkl",
"tuesday_unstemmed_highN_WSDdata_64kmeans.pkl",
"tuesday_unstemmed_highN_WSDdata_64ward.pkl"]

# SWITCH TO THIS AFTER!
next_batch=["monday_unstemmed_WSDdata_sage_kmeans.pkl",
"tuesday_unstemmed_highN_WSDdata_64ward.pkl"]


final_batch=["monday_unstemmed_WSDdata_sage_kmeans.pkl",
"monday_unstemmed_WSDdata_sage_ward.pkl",
"monday_unstemmed_WSDdata_sage_cos.pkl",
"monday_unstemmed_WSDdata_sage_affinity.pkl"]

newfinalbatch=[
"tuesday_unstemmed_BOTHhighthresh_WSDdata_sage_kmeans.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_sage_ward.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_sage_cos.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_sage_affinity.pkl",
"tuesday_unstemmed_highN_WSDdata_sage_affinity.pkl",
"tuesday_unstemmed_highN_WSDdata_sage_cos.pkl",
"tuesday_unstemmed_highN_WSDdata_sage_kmeans.pkl",
"tuesday_unstemmed_highN_WSDdata_sage_ward.pkl",

"monday_unstemmed_WSDdata_128affinity.pkl",
"monday_unstemmed_WSDdata_128cos.pkl",
"monday_unstemmed_WSDdata_128kmeans.pkl",
"monday_unstemmed_WSDdata_128ward.pkl",

"tuesday_unstemmed_BOTHhighthresh_WSDdata_128affinity.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_128cos.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_128kmeans.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_128ward.pkl",

"tuesday_unstemmed_highN_WSDdata_128affinity.pkl",
"tuesday_unstemmed_highN_WSDdata_128cos.pkl",
"tuesday_unstemmed_highN_WSDdata_128kmeans.pkl",
"tuesday_unstemmed_highN_WSDdata_128ward.pkl",

"monday_unstemmed_WSDdata_sage_dbscan.pkl",
"tuesday_unstemmed_BOTHhighthresh_WSDdata_sage_dbscan.pkl",
"tuesday_unstemmed_highN_WSDdata_sage_dbscan.pkl"
]


all_cluster_files=newfinalbatch
numberoffiles=len(all_cluster_files)
filenum=1
for clusters_file_name in all_cluster_files:
	start_time=time.time()
	# UNSUPERVISED EVALUATION
	# nouns

	to_open="dumps/"+clusters_file_name
	with open(to_open,'r') as f:
		pre_dict=cPickle.load(f)

	embedding_mat=pre_dict["embeds"]
	targets=pre_dict["targets"]
	allwords=pre_dict["allwords"]
	target2senses=pre_dict["target2senses"]
	node2ix=pre_dict["node2ix"]
	words_with_embeddings=set(node2ix.keys())

	embedding_dim=len(embedding_mat[0,:])

	save_file_name="output_"+clusters_file_name[:len(clusters_file_name)-4]+".txt"
	full_save_file_name="dumps/outputs/"+save_file_name

	outfile=open(full_save_file_name,"w+")
	path = "./SemEval-2010/test_data/nouns/"

	for filename in os.listdir(path):
		print(filename)
		fl=path+filename
		word=filename.split('.',1)[0]
		with open(fl,'r') as f:
			soup=BeautifulSoup(f,"html5lib")
			i=1
			for section in soup.body.contents[0].contents:
				testing_instance=word+'.n.'+str(i)
				i=i+1
				context=section.text;
				context=process_sentence_e(0,context)
				context_embedding=np.zeros(embedding_dim);
				counter=0
				totalwordsincontext=0;
				for cword in context:
					totalwordsincontext=totalwordsincontext+1;
					if cword in words_with_embeddings and cword not in targets:
						context_embedding=context_embedding+embedding_mat[node2ix[cword],:]
						counter=counter+1 
				print('Progress on cluster files: '+str((filenum+0.0)/numberoffiles)+' Fraction of words found: '+str((counter+0.0)/totalwordsincontext))
				if counter!=0:
					context_embedding=context_embedding/counter
				# find closest sense embedding to the context
				best_sense_ind=closest_sense(context_embedding,target2senses[word][0],"cos")

				print('Cluster sizes: '),
				print(np.unique(target2senses[word][3], return_counts=True)[1])

				print('Closest cluster index: '+str(best_sense_ind))
				ln=word+".n "+testing_instance+" "+word+".n."+str(best_sense_ind)
				ln=ln+"\n"
				outfile.write(ln)
				print('Time elapsed on this file: '+str((time.time()-start_time)/60.0))

	# verbs

	path = "./SemEval-2010/test_data/verbs/"


	for filename in os.listdir(path):
		print(filename)
		fl=path+filename
		word=filename.split('.',1)[0]
		with open(fl,'r') as f:
			soup=BeautifulSoup(f,"html5lib")
			i=1
			for section in soup.body.contents[0].contents:
				testing_instance=word+'.v.'+str(i)
				i=i+1
				context=section.text;
				context=process_sentence_e(0,context)
				context_embedding=np.zeros(embedding_dim);
				counter=0
				totalwordsincontext=0;
				for cword in context:
					totalwordsincontext=totalwordsincontext+1;
					if cword in words_with_embeddings and cword not in targets:
						context_embedding=context_embedding+embedding_mat[node2ix[cword],:]
						counter=counter+1
				print('Progress on cluster files: '+str((filenum+0.0)/numberoffiles)+' Fraction of words found: '+str((counter+0.0)/totalwordsincontext))
				if counter!=0:
					context_embedding=context_embedding/counter
				# find closest sense embedding to the context
				best_sense_ind=closest_sense(context_embedding,target2senses[word][0],"cos")

				print('Cluster sizes: '),
				print(np.unique(target2senses[word][3], return_counts=True)[1])

				print('Closest cluster index: '+str(best_sense_ind))
				ln=word+".v "+testing_instance+" "+word+".v."+str(best_sense_ind)
				ln=ln+"\n"
				outfile.write(ln)
				print('Time elapsed on this file: '+str((time.time()-start_time)/60.0))
	filenum=filenum+1

		# later: can make it smaller window

# how much of a window should the context vector have??

