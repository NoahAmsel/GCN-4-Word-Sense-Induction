# CACooN:
## (Clustering Augmented Cooccurrence Network for word sense embeddings)

Hannah Lawrence & Noah Amsel

This repository contains code for our final project in CPSC 667: Advanced Natural Language Processing.

Each script contains a description inside its `main` function. Run

```
python cooccurrence.py
python sage.py
python induce.py
python evaluate.py
```

### Dependencies
You'll need `nltk`, `networkx`, `numpy`, and possibly others.  
The code uses node2vec and GraphSAGE. You will need to clone those repositories and place them in the same parent directory that this repository is in. (Note that we use the Veles implementation of node2vec due to the high memory requirements of the original version)
```
https://github.com/williamleif/GraphSAGE
https://github.com/vid-koci/snap/tree/veles/examples/veles
```
For size reasons, some files can't be uploaded to GitHub:
We use pretrained Glove embeddings, which are available for download [here](https://nlp.stanford.edu/projects/glove/). Put them in a folder inside this repository called `glove`.
Finally, we use data from the SemEval-2010 task, available for download [here](https://www.cs.york.ac.uk/semeval2010_WSI/datasets.html). Put them in a folder called `SemEval-2010` with subfolders `evaluation`, `test_data`, and `training_data`,

There are two hardcoded file paths in the `sage.py` script but they should be easy to spot and change if necessary.
