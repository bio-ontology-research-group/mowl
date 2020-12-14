import numpy as np
from owlready2 import get_ontology


class Dataset(object):
    def __init__(self, url, split_ratio = (0.6,0.2,0.2), *args, **kwargs):
        self.url = url
        self._loaded = False
        self.split_ratio = split_ratio

    def _load(self):
        raise NotImplementedError()


    @property
    def train_set(self):
        if not self._loaded:
            self._load()
        return self._training
    
    
    @property
    def test_set(self):
        if not self._loaded:
            self._load()
        return self._testing
    
    @property
    def valid_set(self):
        if not self._loaded:
            self._load()
        return self._validation
    

    @classmethod
    def owl2ntriples(source_file, target_file):
        onto = get_ontology(source_file).load()
        onto.save(file = target_file, format = 'ntriples')

    @classmethod
    def split_data(data, ratio):
        raise NotImplementedError()

    def negative_samples(self):
        raise NotImplementedError()


#TODO: Classes for handling remote (tar/zip) and local files