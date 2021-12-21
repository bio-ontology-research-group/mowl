from mowl.walking.walking import WalkingModel
import networkx as nx
import numpy as np
import random
import logging
from multiprocessing import Pool, get_context

class Node2Vec(WalkingModel):
    
    '''
    Reference implementation of node2vec. 

    :param p: Return hyperparameter. Default is 1.
    :type p: float
    :param q: Input hyperparameter. Default is 1.
    :type p: float

    Original author: Aditya Grover

    For more details, refer to the paper:
    node2vec: Scalable Feature Learning for Networks
    Aditya Grover and Jure Leskovec 
    Knowledge Discovery and Data Mining (KDD), 2016

    Adapted by Sonja Katz, 2021
    Readapted to subclass WalkingModel abstract class. By F. Zhapa 
    '''

    def __init__(self, 
                 edges, 
		 num_walks, 
		 walk_length, 
		 p, 
		 q,
                 num_workers=1,
                 seed = 0,
		 outfile = None): 
        
        super().__init__(edges, num_walks, walk_length, num_workers, outfile) 

        self.p = p
        self.q = q

        # self.is_directed = is_directed
	# self.is_weighted = is_weighted
	# self.embeddings_file_path=f"{self.data_root}/{embeddings_file_path}"

        self.graph = nx.Graph()
        
        for edge in edges:
            src, rel, dst, weight = edge.src(), edge.rel(), edge.dst(), edge.weight()
            self.graph.add_edge(src, dst)
            self.graph.edges[src,dst]["type"] = rel
            self.graph.edges[src,dst]["weight"] = weight

        self.nodes = list(self.graph.nodes())


    def walk(self):
                
        paths_per_worker = self.num_paths_per_worker()

        self.preprocess_transition_probs(G)
    
        file_names = [f"tmpfile_{i}.txt" for i in range(self.num_workers)]
        logging.debug("FILENAMES %s", str(file_names))

        args_list = []
        for i in range(self.num_workers):
            args_list.append((paths_per_worker[i], self.walk_length, file_names[i]))
            
        
        logging.debug("Starting Pool")

        with get_context('spawn').Pool(processes=self.num_workers) as pool:
            res = pool.map(self.write_walks_to_disk, args_list)

        
        #combine_files
        with open(self.outfile, 'w') as finalout:
            for file_ in file_names:
                with open(file_, 'r') as f:
                    for line in f:
                        finalout.write(f"{line}\n")



    def write_walks_to_disk(self, args):
    
        num_walks, walk_length, out_file = args
        
        with open(out_file, 'w')  as fout:
            for walk in self.simulate_walks(num_walks, walk_length):
                fout.write(u"{}\n".format(u" ".join(v for v in walk)))

        return out_file
        
                
	

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        nodes = self.nodes[:]
        
        for i in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                yield(self.node2vec_walk(walk_length=walk_length, start_node=node))


    
    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(self.graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next_ = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next_)
            else:
                break

        return [str(node) for node in walk]


    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''

        alias_nodes = {}
        for node in self.nodes:
            unnormalized_probs = [self.graph[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if self.is_directed:
            for edge in self.graph.edges():
                alias_edges[edge] = self.get_alias_edge(self.graph, edge[0], edge[1])
        else:
            for edge in self.graph.edges():
                alias_edges[edge] = self.get_alias_edge(self.graph, edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(self.graph, edge[1], edge[0])

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


