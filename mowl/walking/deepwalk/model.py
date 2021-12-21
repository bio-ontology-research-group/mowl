
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
    
    def __init__(self, edges, num_paths, path_length, alpha, num_workers=1, seed = 0, outfile=None):
        super().__init__(edges, num_paths, path_length, alpha, num_workers, outfile)

        self.graph = nx.Graph()
        self.rand = random.Random(seed)
        
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

        if self.num_paths <= self.num_workers:
            paths_per_worker = [1 for x in range(self.num_paths)]
        else:
            remainder = self.num_paths % self.num_workers
            aux = self.num_workers - remainder
            paths_per_worker = [int((self.num_paths+aux)/self.num_workers) for i in range(self.num_workers)]
            i = 0
            while aux > 0:
                paths_per_worker[i%self.num_workers] -= 1
                i += 1
                aux -= 1

            logging.debug("PATHS PER WORKER %s", str(paths_per_worker))


        file_names = [f"tmpfile_{i}.txt" for i in range(self.num_workers)]
        logging.debug("FILENAMES %s", str(file_names))

        args_list = []
        for i in range(self.num_workers):
            args_list.append((paths_per_worker[i], self.path_length, self.alpha, random.Random(self.rand.randint(0, 2**31)), file_names[i]))

        files = []

        logging.debug("Starting Pool")

        with get_context('spawn').Pool(processes=self.num_workers) as pool:
            res = pool.map(self.write_walks_to_disk, args_list)

        
#        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
#            executor.map(self.write_walks_to_disk, args_list)
            #    files.append(file_)
        #combine_files
        with open(self.outfile, 'w') as finalout:
            for file_ in file_names:
                with open(file_, 'r') as f:
                    for line in f:
                        finalout.write(f"{line}\n")


                

    def write_walks_to_disk(self, args):
        
        current_graph = deepcopy(self.graph)

        num_paths, path_length, alpha, rand, out_file = args
        
        with open(out_file, 'w')  as fout:
            for walk in self.build_deepwalk_corpus(current_graph, num_paths, path_length, alpha, rand):
                fout.write(u"{}\n".format(u" ".join(v for v in walk)))

        return out_file
        

            
    def build_deepwalk_corpus(self, graph, num_paths, path_length, alpha, rand=random.Random(0)):
        nodes = list(graph.nodes)

        for i in range(num_paths):
            rand.shuffle(nodes)
            for node in nodes:
                yield(self.random_walk(path_length, rand=rand, alpha=alpha, start=node))


    def random_walk(self, path_length, alpha=0, rand = random.Random(), start=None):

        '''
        
        :param path_length: Length of the random walk.
        :param alpha: probability of restarts.
        :param start: the start node of the random walk.

        :return: Returns a truncated random walk.
        '''

        graph = self.graph

        if start:
            path = [start]
        else:
            path = [rand.choice(self.nodes)]

        while len(path) < path_length:
            currNode = path[-1]
            neighbors = list(graph.neighbors(currNode))
            if len(neighbors) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(neighbors))
                else:
                    path.append(path[0])
            else:
                break

        return [str(node) for node in path]
