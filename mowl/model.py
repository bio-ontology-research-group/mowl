class Model(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self._testing_set = None
    
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
        

    def get_embeddings_data(self):
        raise NotImplementedError()
