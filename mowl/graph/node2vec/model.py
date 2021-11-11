from mowl.model import Model
import networkx as nx
import numpy as np
import random
from gensim.models import Word2Vec

class Node2Vec():
	'''
	Reference implementation of node2vec. 

	Original author: Aditya Grover

	For more details, refer to the paper:
	node2vec: Scalable Feature Learning for Networks
	Aditya Grover and Jure Leskovec 
	Knowledge Discovery and Data Mining (KDD), 2016

	Adapted by Sonja Katz, 2021
	'''

	def __init__(self, 
				edgelist, 
				p, 
				q, 
				num_walks, 
				walk_length, 
				embeddings_file_path,
				dimensions=128, 
				window_size=10, 
				workers=8, 
				iter=1, 
				is_directed=False, 
				is_weighted=False, 
				data_root = "."): 

		#super().__init__(dataset) 
		self.data_root = data_root
		self.edgelist = edgelist
		self.p = p
		self.q = q
		self.num_walks = num_walks
		self.walk_length = walk_length
		self.dimensions = dimensions
		self.window_size = window_size
		self.workers = workers
		self.iter = iter
		self.is_directed = is_directed
		self.is_weighted = is_weighted
		self.embeddings_file_path=f"{self.data_root}/{embeddings_file_path}"
		
	def read_graph(self):
		'''
		Reads the input network in networkx.
		'''

		if self.is_weighted:
			G = nx.read_edgelist(self.edgelist, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
		else:
			G = nx.read_edgelist(self.edgelist, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

		if not self.is_directed:
			G = G.to_undirected()

		self.G = G

		return G

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''

		G = self.G

		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def learn_embeddings(self, walks):
		'''
		Learn embeddings by optimizing the Skipgram objective using SGD.
		'''
		walks = [list(map(str, walk)) for walk in walks]
		model = Word2Vec(walks, window=self.window_size, vector_size=self.dimensions, min_count=0, sg=1, workers=self.workers, epochs=self.iter)
		model.wv.save_word2vec_format(self.embeddings_file_path)
	
		return

	def preprocess_transition_probs(self, G):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''

		#G = self.G

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = self.alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if self.is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(G, edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(G, edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(G, edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


	def alias_setup(self, probs):
		'''
		Compute utility lists for non-uniform sampling from discrete distributions.
		Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
		for details
		'''
		K = len(probs)
		q = np.zeros(K)
		J = np.zeros(K, dtype=int)

		smaller = []
		larger = []
		for kk, prob in enumerate(probs):
			q[kk] = K*prob
			if q[kk] < 1.0:
				smaller.append(kk)
			else:
				larger.append(kk)

		while len(smaller) > 0 and len(larger) > 0:
			small = smaller.pop()
			large = larger.pop()

			J[small] = large
			q[large] = q[large] + q[small] - 1.0
			if q[large] < 1.0:
				smaller.append(large)
			else:
				larger.append(large)

		return J, q


	def get_alias_edge(self, G, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/self.p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/self.q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return self.alias_setup(normalized_probs)

	def alias_draw(self, J, q):
		'''
		Draw sample from a non-uniform discrete distribution using alias sampling.
		'''
		K = len(J)

		kk = int(np.floor(np.random.rand()*K))
		if np.random.rand() < q[kk]:
			return kk
		else:
			return J[kk]

	
	def train(self):
		G = self.read_graph()
		self.preprocess_transition_probs(G)
		walks = self.simulate_walks(self.num_walks, self.walk_length)
		self.learn_embeddings(walks)
		return 
	
