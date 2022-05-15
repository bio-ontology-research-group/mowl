class Model(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self._testing_set = None

                

    @property
    def testing_set(self):
        return self._testing_set

    @property
    def training_set(self):
        return self._training_set

    @property
    def head_entities(self):
        return self._head_entities
    @property
    def tail_entities(self):
        return self._tail_entities
    
    def train(self):
        '''Trains the model
        '''
        raise NotImplementedError()

    def eval_method():
        raise NotImplementedError()
    
    def evaluate(self):
        '''Evaluates the model
        '''
        raise NotImplementedError()


class EmbeddingModel(Model):

    def __init__(self, dataset):
        super().__init__(dataset)


    def get_entities_index_dict(self):
        return self.class_index_dict, self.relation_index_dict
        
