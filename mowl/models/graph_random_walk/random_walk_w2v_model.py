from mowl.base_models.graph_model import RandomWalkModel
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import mowl.error.messages as msg
import os
import time
from deprecated.sphinx import versionadded

@versionadded(version="0.2.0")
class RandomWalkPlusW2VModel(RandomWalkModel):
    """
    Embedding model that combines graph projections + random walks.
    """
    
    def __init__(self, *args, **kwargs):
        super(RandomWalkPlusW2VModel, self).__init__(*args, **kwargs)

        self._edges = None
        self.w2v_model = None
        self.update_w2v_model = False
        self.axioms_added = False

    @property
    def class_embeddings(self):
        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if len(self.w2v_model.wv) == 0:
            raise AttributeError(msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND)
        
        cls_embeds = {}
        for cls in self.dataset.classes.as_str:
            if cls in self.w2v_model.wv:
                cls_embeds[cls] = self.w2v_model.wv[cls]
        return cls_embeds

    @property
    def object_property_embeddings(self):
        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if len(self.w2v_model.wv) == 0:
            raise AttributeError(msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND)

        obj_prop_embeds = {}
        for obj_prop in self.dataset.object_properties.as_str:
            if obj_prop in self.w2v_model.wv:
                obj_prop_embeds[obj_prop] = self.w2v_model.wv[obj_prop]
        return obj_prop_embeds

    @property
    def individual_embeddings(self):
        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if len(self.w2v_model.wv) == 0:
            raise AttributeError(msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND)
        
        
        ind_embeds = {}
        for ind in self.dataset.individuals.as_str:
            if ind in self.w2v_model.wv:
                obj_prop_embeds[ind] = self.w2v_model.wv[ind]
        return ind_embeds

    
    def set_w2v_model(self, *args, **kwargs):
        """
        Sets the Word2Vec model to be used for training. The set model will be a :class:`gensim.models.word2vec.Word2Vec` object.

        :param args: Arguments to be passed to the :class:`Word2Vec <gensim.models.word2vec.Word2Vec>` constructor.
        :param kwargs: Keyword arguments to be passed to the :class:`Word2Vec <gensim.models.word2vec.Word2Vec>` constructor.
        """
        self.w2v_model = Word2Vec(*args, **kwargs)

    def train(self, epochs=None):
        """
        Triggers the Word2Vec training process.

        :param epochs: Number of epochs to train the model. If None, the value of the epochs parameter passed to the constructor will be used.
        :type epochs: int
        """
        if self.projector is None:
            raise AttributeError(msg.GRAPH_MODEL_PROJECTOR_NOT_SET)
        if self.walker is None:
            raise AttributeError(msg.RANDOM_WALK_MODEL_WALKER_NOT_SET)
        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if epochs is None:
            epochs = self.w2v_model.epochs

        if self._edges is None or self.axioms_added:
            self.axioms_added = False
            self._edges = self.projector.project(self.dataset.ontology)
            self.walker.walk(self._edges)
            

            # This loop is needed to make sure the file is written to disk before running Word2Vec
            last_modified = os.path.getmtime(self.walker.outfile)
            while True:
                current_modified = os.path.getmtime(self.walker.outfile)
                if current_modified != last_modified:
                    break
                time.sleep(0.1)
                last_modified = current_modified
            
        sentences = LineSentence(self.walker.outfile)
        self.w2v_model.build_vocab(sentences, update=self.update_w2v_model)
        if epochs > 0:
            self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=epochs)
        
    def add_axioms(self, *axioms):
        classes = set()
        object_properties = set()
        individuals = set()

        for axiom in axioms:
            classes |= set(axiom.getClassesInSignature())
            object_properties |= set(axiom.getObjectPropertiesInSignature())
            individuals |= set(axiom.getIndividualsInSignature())

        new_entities = list(classes.union(object_properties).union(individuals))
            
        self.dataset.add_axioms(*axioms)
        self._edges = self.projector.project(self.dataset.ontology)
        self.walker.walk(self._edges, nodes_of_interest=new_entities)
        self.update_w2v_model = True
        #Rebuild vocab
        sentences = LineSentence(self.walker.outfile)
        self.w2v_model.build_vocab(sentences, update=self.update_w2v_model)
        self.axioms_added = True
        
    def from_pretrained(self, model):
        
        if not isinstance(model, str):
            raise TypeError("Parameter model must be a string pointing to the Word2Vec model file.")

        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")
        
        self._is_pretrained = True
        if not isinstance(model, str):
            raise TypeError

        self.w2v_model = Word2Vec.load(model)
    

