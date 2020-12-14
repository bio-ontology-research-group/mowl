from datasets.base import Dataset
import math
import random
import numpy as np
import gzip

class PPI_String(Dataset):
    def __init__(self, url, *args, **kwargs):
        super(PPI_String, self).__init__(url, *args, **kwargs)
        self._annotations_loaded = False
        self._mappings_loaded = False

    def _load(self, score_threshold = 700):
        if self._loaded:
            return

        self.interactions = {}    
        self.data = []    
            
        with open(self.url, 'rt') as f:
            next(f)
            for line in f:
                p1, p2, score = line.strip().split()
                if float(score) < score_threshold:
                    continue
                if p1 not in self.interactions:
                    self.interactions[p1] = set()
                if p2 not in self.interactions:
                    self.interactions[p2] = set()
                if p2 not in self.interactions[p1]:
                    self.interactions[p1].add(p2)
                    self.interactions[p2].add(p1)
                    self.data.append((p1, p2))

        self._loaded = True

    @classmethod
    def split(data, ratio = (0.6,0.2,0.2)):
        
        if sum(ratio) != 1.0:
            raise Exception("Invalid split ratio")
            
        np.random.shuffle(data)
        n_data = len(data)
        train_n = int(math.ceil(n_data*ratio[0]))
        test_n = int(math.ceil(n_data*ratio[1]))
        
        return data[:train_n], data[train_n:train_n + test_n], data[train_n + test_n:]


    def negative_samples(self):
        proteins = set()
        negatives = []

        for (p1,p2) in self.data:
            proteins.add(p1)
            proteins.add(p2)

        while len(negatives) < len(self.data):
            sample = random.sample(proteins, 2)
            p1 = sample[0]
            p2 = sample[1]
            if (p1, p2) in negatives or (p2, p1) in negatives:
                continue
            if p1 not in self.interactions[p2]:
                negatives.append((p1, p2))
                
        self.neg_train_data, self.neg_test_data, self.neg_valid_data = split(negatives, self.split_ratio)


    def load_mappings(self, path):
        self.mapping = {}
        
        with gzip.open(path, 'rt') as f:
            next(f)
            for line in f:
                string_id, p_id, sources = line.strip().split('\t')
                if self.source  not in sources:
                    continue
                if p_id not in self.mapping:
                    self.mapping[p_id] = set()
                self.mapping[p_id].add(string_id)

        self._mappings_loaded = True

    def load_annotations(self, gaf_file):
        self.annotations = set()
        
        with gzip.open(gaf_file, 'rt', encoding ='utf-8') as f:
            for line in f:
                if line.startswith('!'):
                    continue
                it = line.strip().split('\t')
                p_id = it[1]
                go_id = it[4]
                if it[6] == 'IEA' or it[6] == 'ND':
                    continue
                if p_id not in self.mapping:
                    continue
                s_ids = self.mapping[p_id]
                for s_id in s_ids:
                    self.annotations.add((s_id, go_id))
                    
        self._annotations_loaded = True

    def plain_data(self, obo_file):
        tdf = open(f'test.plain.nt', 'w')
        # Load GO
        with open(obo_file) as f:
            tid = ''
            for line in f:
                line = line.strip()
                if line.startswith('id:'):
                    tid = line[4:]
                if not tid.startswith('GO:'):
                    continue
                if line.startswith('is_a:'):
                    tid2 = line[6:].split(' ! ')[0]
                    tdf.write(f'<http://{tid}> <http://is_a> <http://{tid2}> .\n')
                if line.startswith('relationship:'):
                    it = line[14:].split(' ! ')[0].split()
                    tdf.write(f'<http://{tid}> <http://{it[0]}> <http://{it[1]}> .\n')

        # Load interactions
        for (p1, p2) in self.data:
            tdf.write(f'<http://{p1}> <http://interacts> <http://{p2}> .\n')

        # Load annotations
        for (p_id, go_id) in self.annotations:
            tdf.write(f'<http://{p_id}> <http://hasFunction> <http://{go_id}> .\n')
                
        tdf.close()

