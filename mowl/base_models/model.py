from deprecated.sphinx import deprecated, versionchanged
import tempfile

@versionchanged(version = "0.1.0", reason = "Parameter ``model_filepath`` added in the base class for all models. Optional parameter that will use temporary files in case it is not set.")
class Model(object):
    def __init__(self, dataset, model_filepath = None):
        self.dataset = dataset
        self._model_filepath = model_filepath
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
    def model_filepath(self):
        if self._model_filepath is None:
            model_filepath = tempfile.NamedTemporaryFile()
            self._model_filepath = model_filepath.name
        return self._model_filepath
        
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

    def __init__(self, dataset, model_filepath = None):
        super().__init__(dataset, model_filepath = model_filepath)
        
    @deprecated(version = "0.1.0")
    def get_entities_index_dict(self):
        raise NotImplementedError()
        

    def get_embeddings_data(self):
        raise NotImplementedError()



