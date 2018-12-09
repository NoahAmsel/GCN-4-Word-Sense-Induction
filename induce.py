import os

class Induce():
    def node2vec(input, output):
        node2vec_path = "../snap/examples/node2vec/node2vec"
        dims = 128
        walklength = 10
        walkspersource = 10
        contextsize = 5 #make it 5?
        p = 1.0     # lower = likely to backtrack
        q = 1.0     # lower = explore whole community


        comm = ("{bin} -i:{input} -o:{output} -d:{dims} " +
                "-l:{walklength} -r:{walkspersource} " +
                "-k:{contextsize} -p:{p} -q:{q} -w").format(bin=node2vec_path, input=input,
                                                          output=output, dims=dims, walklength=walklength,
                                                          walkspersource=walkspersource, contextsize=contextsize,
                                                          p=p, q=q)
        print(comm)
        status = os.system(comm)

    def read_embs(emb_file, labels):
        with open(emb_file, "r") as f:
            line = f.readline()
            words, dims = line.split()[:2]
            self.embeds = np.zeros((int(words),int(dims)))
            line = f.readline()
            while line:
                sp = line.split()
                row = int(sp[0])
                self.embeds[row,:] = [float(s) for s in sp[1:]]
                line = f.readline()

        self.word2ix = {}
        with open(labels, "r") as f:
            for line in f:
                ix, word = line.split()[:2]
                self.word2ix[word] = ix

        return embeds, word2ix

    def induce_word(word, targets2edges):


if __name__ == "__main__":
    node2vec("dumps/sun1am", "dumps/sun3am.embeddings")
