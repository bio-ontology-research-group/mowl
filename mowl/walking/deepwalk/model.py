
import networkx as nx
from mowl.walking.walking import WalkingModel
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, get_context
import logging
from copy import deepcopy

logging.basicConfig(level=logging.INFO)


class DeepWalk(WalkingModel):

    '''
    Implementation of DeepWalk based on <https://github.com/phanein/deepwalk/blob/master/deepwalk/graph.py>
    '''
    
    def __init__(self,
                 edges,
                 num_walks,
                 walk_length,
                 alpha,
                 num_workers=1,
                 seed = 0,
                 outfile=None):

        super().__init__(edges, num_walks, walk_length, alpha, num_workers, outfile)

        self.graph = nx.Graph()
        self.rand = random.Random(seed)
        self.alpha = alpha
        
        for edge in edges:
            src, rel, dst = edge.src(), edge.rel(), edge.dst()
            
            self.graph.add_edge(src, dst)
            self.graph.edges[src,dst]["type"] = rel
            self.graph.nodes[src]["val"] = False
            self.graph.nodes[dst]["val"] = False

        
        self.nodes = list(self.graph.nodes())
        
        
    def walk(self):

        '''
        Implementation of parallelizable DeepWalk.
        '''

        paths_per_worker = self.num_paths_per_worker()       

        file_names = [f"tmpfile_{i}.txt" for i in range(self.num_workers)]
        logging.debug("FILENAMES %s", str(file_names))

        args_list = []
        for i in range(self.num_workers):
            args_list.append((paths_per_worker[i], self.walk_length, random.Random(self.rand.randint(0, 2**31)), file_names[i]))

        files = []

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
        
        num_walks, walk_length, rand, out_file = args
        
        with open(out_file, 'w')  as fout:
            for walk in self.build_deepwalk_corpus(num_walks, walk_length, rand):
                fout.write(u"{}\n".format(u" ".join(v for v in walk)))

        return out_file
        

            
    def build_deepwalk_corpus(self, num_walks, walk_length, rand=random.Random(0)):
        nodes = self.nodes[:]

        for i in range(num_walks):
            rand.shuffle(nodes)
            for node in nodes:
                yield(self.random_walk(walk_length, rand=rand, start=node))


    def random_walk(self, walk_length, rand = random.Random(), start=None):

        '''
        
        :param walk_length: Length of the random walk.
        :param alpha: probability of restarts.
        :param start: the start node of the random walk.

        :return: Returns a truncated random walk.
        '''

        graph = self.graph

        if start:
            walk = [start]
        else:
            walk = [rand.choice(self.nodes)]

        while len(walk) < walk_length:
            currNode = walk[-1]
            neighbors = list(graph.neighbors(currNode))
            if len(neighbors) > 0:
                if rand.random() >= self.alpha:
                    walk.append(rand.choice(neighbors))
                else:
                    walk.append(walk[0])
            else:
                break

        return [str(node) for node in walk]
