from deprecated.sphinx import deprecated, versionchanged, versionadded
import tempfile
from mowl.datasets import Dataset
from mowl.owlapi import OWLAPIAdapter
from java.util import HashSet



@versionchanged(version="0.1.0", reason="Parameter ``model_filepath`` added in the base class for \
    all models. Optional parameter that will use temporary files in case it is not set.")
class Model():
    def __init__(self, dataset, model_filepath=None):
        """Abstract model class
        :param dataset: Dataset object.
        :type dataset: mowl.datasets.base.Dataset
        :param model_filepath: Path for saving the model. Defaults to a temporary file path.
        :type model_filepath: str, optional
        """

        if not isinstance(dataset, Dataset):
            raise TypeError("Parameter dataset must be a mOWL Dataset.")

        if model_filepath is not None and not isinstance(model_filepath, str):
            raise TypeError("Optional parameter model_filepath must be of type str.")

        self.dataset = dataset
        self._model_filepath = model_filepath
        self._testing_set = None
                        

    def train(self, *args, **kwargs):
        '''Abstract method for training the model. This method must be implemented in children classes
        '''
        raise NotImplementedError("Method train is not implemented.")

    def eval_fn(self, *args, **kwargs):
        raise NotImplementedError("Method eval_fn is not implemented.")

    @versionadded(version="0.2.0", reason="Axiom scoring method added to the base class.")
    def score(self, axiom):
        """
        Returns the score of the given axiom.

        :param axiom: The axiom to score.
        :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
        """
        
        raise NotImplementedError("Method score is not implemented.")

    
    @property
    def model_filepath(self):
        """Path for saving the model.

        :rtype: str
        """
        if self._model_filepath is None:
            model_filepath = tempfile.NamedTemporaryFile()
            self._model_filepath = model_filepath.name
        return self._model_filepath

    @property
    def class_index_dict(self):
        """Dictionary with class names as keys and class indexes as values.

        :rtype: dict
        """
        class_to_id = {v: k for k, v in enumerate(self.dataset.classes.as_str)}
        return class_to_id

    @property
    def individual_index_dict(self):
        """Dictionary with individual names as keys and indexes as values.

        :rtype: dict
        """
        individual_to_id = {v: k for k, v in enumerate(self.dataset.individuals.as_str)}
        return individual_to_id
                            
    @property
    def object_property_index_dict(self):
        """Dictionary with object property names as keys and object property indexes as values.

        :rtype: dict
        """
        object_property_to_id = {v: k for k, v in enumerate(self.dataset.object_properties.as_str)}
        return object_property_to_id

    @versionadded(version="0.2.0")
    @property
    def class_embeddings(self):
        """
        Returns a dictionary with class names as keys and class embeddings as values.
        
        :rtype: dict
        """
        raise NotImplementedError()

    @versionadded(version="0.2.0")
    @property
    def object_property_embeddings(self):
        """
        Returns a dictionary with object property names as keys and object property embeddings as values.

        :rtype: dict
        """
        raise NotImplementedError()

    @versionadded(version="0.2.0")
    @property
    def individual_embeddings(self):
        """
        Returns a dictionary with individual names as keys and individual embeddings as values.

        :rtype: dict
        """
        raise NotImplementedError()

    @versionadded(version="0.2.0")
    def add_axioms(self, *axioms):
        """
        This method adds axioms to the dataset contained in the model and reorders the embedding information for each entity accordingly. New entites are initalized with random embedding.
        
        :param axioms: Axioms to be added to the dataset.
        :type axioms: org.semanticweb.owlapi.model.OWLAxiom
        """
        raise NotImplementedError()

    @versionadded(version="0.2.0")
    def from_pretrained(self, file_name):
        """
        This method loads a pretrained model from a file.

        :param file_name: Path to the pretrained model file.
        :type file_name: str
        """
        raise NotImplementedError()
    
class EmbeddingModel(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_embeddings_data(self):
        raise NotImplementedError()
