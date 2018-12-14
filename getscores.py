import os
import numpy as np
#import pretrained
import glob
import cPickle
import json
#import cooccurrence as co
from scipy import spatial
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from bs4 import BeautifulSoup
import nltk
import time
#import cooccurrence

# Use tagged testing data to run SemEval's evaluation scripts for F-Score/V-Measure on all files
# in the dumps/output folder. Also, rerun evaluation scripts on provided baselines
# Run from the unsupervised evaluation folder
# Run the Fscore/Vmeasure evaluation scripts on every tagged file in outputs folder

# Run all baselines first
print("------------BASELINES-----------")
print("F-Score")
print('--- mfs_all.key ---')
os.system("java -jar fscore.jar ./baselines/mfs_all.key ./keys/all.key all | grep 'Total FScore'")
print('--- 1cl1inst.key ---')
os.system("java -jar fscore.jar ./baselines/1cl1inst.key ./keys/all.key all | grep 'Total FScore'")
print('--- random1.key ---')
os.system("java -jar fscore.jar ./baselines/random1.key ./keys/all.key all | grep 'Total FScore'")
print('--- random2.key ---')
os.system("java -jar fscore.jar ./baselines/random2.key ./keys/all.key all | grep 'Total FScore'")
print('--- random3.key ---')
os.system("java -jar fscore.jar ./baselines/random3.key ./keys/all.key all | grep 'Total FScore'")
print('--- random4.key ---')
os.system("java -jar fscore.jar ./baselines/random4.key ./keys/all.key all | grep 'Total FScore'")
print('--- random5.key ---')
os.system("java -jar fscore.jar ./baselines/random5.key ./keys/all.key all | grep 'Total FScore'")

print("V-Measure")
print('--- mfs_all.key ---')
os.system("java -jar vmeasure.jar ./baselines/mfs_all.key ./keys/all.key all | grep 'Total V-Measure'")
print('--- 1cl1inst.key ---')
os.system("java -jar vmeasure.jar ./baselines/1cl1inst.key ./keys/all.key all | grep 'Total V-Measure'")
print('--- random1.key ---')
os.system("java -jar vmeasure.jar ./baselines/random1.key ./keys/all.key all | grep 'Total V-Measure'")
print('--- random2.key ---')
os.system("java -jar vmeasure.jar ./baselines/random2.key ./keys/all.key all | grep 'Total V-Measure'")
print('--- random3.key ---')
os.system("java -jar vmeasure.jar ./baselines/random3.key ./keys/all.key all | grep 'Total V-Measure'")
print('--- random4.key ---')
os.system("java -jar vmeasure.jar ./baselines/random4.key ./keys/all.key all | grep 'Total V-Measure'")
print('--- random5.key ---')
os.system("java -jar vmeasure.jar ./baselines/random5.key ./keys/all.key all | grep 'Total V-Measure'")


outputdir="/home/lily/hll5/finalproject/GCN-4-Word-Sense-Induction/dumps/outputs/"
for filename in os.listdir(outputdir):
	print("****************** "+filename+" ******************")
	os.system("java -jar fscore.jar "+ outputdir+filename+" ./keys/all.key all | grep 'Total FScore'")
	os.system("java -jar vmeasure.jar "+outputdir+filename+" ./keys/all.key all | grep 'Total V-Measure'")
	time.sleep(3)
	print("END of "+filename)


