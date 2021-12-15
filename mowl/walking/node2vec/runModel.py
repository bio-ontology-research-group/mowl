import os
from mowl.graph.node2vec.model import Node2Vec

### Generate embedding

Node2Vec("karate.edgelist",
         p=1, 
         q=1,
         num_walks=100, 
         walk_length=10, 
         dimensions=128,  # default:128
		 window_size=10,  # defaut: 10
		 workers=8,     # default: 8
		 iter=1,       # default: 1
		 is_directed=False,     # default: False
		 is_weighted=False,     # default: False
         embeddings_file_path="karate.emb"
).train()



### Optional: tSNE visualisation 

import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# Read embedding
emb = pd.read_csv("karate.emb", skiprows=1, header=None, delim_whitespace=True, index_col=0)
emb = emb.sort_index()

# perform tSNE
X = TSNE(n_components=2, verbose=0, n_iter=250, n_jobs=8).fit_transform(emb)

# Get labels for Karate club
Gk = nx.karate_club_graph()
dic_label=dict()
for i in range(len(Gk.nodes)):
    dic_label[i+1]=Gk.nodes[i]["club"]
emb["label"] = dic_label.values()

# encode colors
color=["red" if i == "Mr. Hi" else "blue" for i in emb["label"]]

# Plot
fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(X[:, 0], X[:, 1], marker='o', c=color)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

