import os
import numpy as np

class ExtractEmbeddings():
	def __init__(self, filename,desiredwords):
		# filename contains the embeddings in form: 1 embedding per line, first token is the word, every other token is entry in vector
		numwords=len(desiredwords)
		found=np.zeros(numwords)
		with open(filename,'r') as f:
			dim=len(f.readline().split(' '))-1;
		with open(filename,'r') as f: # there's probably a better way to do this but don't want it to skip the first line
			embeddings=np.zeros([numwords,dim])
			for line in f:
				wrd=line.split(' ',1)[0]
				print(wrd)
				if wrd in desiredwords:
					ind=desiredwords.index(wrd)
					found[ind]=1
					embeddings[ind,:]=line.split(' ')[1:]
				if sum(found)==numwords:
					break
		self.numwords=numwords
		self.dim=dim
		self.embeddings=embeddings
		self.found=found



if __name__ == "__main__":
	example=ExtractEmbeddings("testembeddings.txt",["the","of","."])
	embed=example.embeddings;
	print(embed[0,0:3])
	print(embed[1,0:3])
	print(embed[2,0:3])
	print(len(embed[0,:]))
	print(len(embed[1,:]))
	print(len(embed[2,:]))
	print(example.found)
