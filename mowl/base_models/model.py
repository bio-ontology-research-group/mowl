from deprecated.sphinx import deprecated

class Model(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self._testing_set = None
        self._class_index_dict = None
        self._object_property_index_dict = None
        
    def train(self):
        '''Trains the model
        '''
        raise NotImplementedError()

    def eval_method(self):
        raise NotImplementedError()
    
    def evaluate(self):
        '''Evaluates the model
        '''
        raise NotImplementedError()

    @property
    def class_index_dict(self):
        if self._class_index_dict is None:
            self._class_index_dict = {v:k for k,v in enumerate(self.dataset.classes)}
        return self._class_index_dict

    @property
    def object_property_index_dict(self):
        if self._object_property_index_dict is None:
            self._object_property_index_dict = {v:k for k,v in enumerate(self.dataset.object_properties)}
        return self._object_property_index_dict

    

class EmbeddingModel(Model):

    def __init__(self, dataset):
        super().__init__(dataset)
        
    @deprecated(version = "0.1.0")
    def get_entities_index_dict(self):
        raise NotImplementedError()
        

    def get_embeddings_data(self):
        raise NotImplementedError()



