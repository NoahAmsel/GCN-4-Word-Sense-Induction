import os

def node2vec():
    node2vec_path = "../snap/examples/node2vec/node2vec"
    input = "dumps/testgraph"
    output = "dumps/embeddings"
    dims = 128
    walklength = 10
    walkspersource = 10
    contextsize = 5 #make it 5?
    p = 1.0     # lower = likely to backtrack
    q = 0.5     # lower = explore whole community


    comm = ("{bin} -i:{input} -o:{output} -d:{dims} " +
            "-l:{walklength} -r:{walkspersource} " +
            "-k:{contextsize} -p:{p} -q:{q} -w").format(bin=node2vec_path, input=input,
                                                      output=output, dims=dims, walklength=walklength,
                                                      walkspersource=walkspersource, contextsize=contextsize,
                                                      p=p, q=q)
    print(comm)
    status = os.system(comm)

node2vec()
