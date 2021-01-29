from dataclasses import dataclass
import numpy as np
from typing import Mapping


@dataclass
class Triples(object):

    filepath: str
    _loaded: False

    def load(self, delimiter='\t', encoding=None):
        if not self._loaded:
            self._triples = np.loadtxt(
                fname=self.filepath,
                dtype=str,
                comments='@Comment@ Head Relation Tail',
                delimiter=delimiter,
                encoding=encoding,
            )
            self._loaded = True
        return self._triples

    

@dataclass
class Dataset(object):

    @property
    def training(self):
        if not self._loaded:
            self._load()
        return self._training
        
    @property
    def testing(self):
        if not self._loaded:
            self._load()
        return self._testing
    
    @property
    def validation(self):
        if not self._loaded:
            self._load()
        return self._validation
    
    @classmethod
    def split_data(data, ratio):
        raise NotImplementedError()

    def negative_samples(self):
        raise NotImplementedError()
