from mowl.base_models import SyntacticModel
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import mowl.error.messages as msg
from deprecated.sphinx import versionadded

@versionadded(version="0.2.0")
class SyntacticPlusW2VModel(SyntacticModel):
    """
    Model that combines corpus generation with Word2Vec training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w2v_model = None
        self.update_w2v_model = False
        self._is_pretrained = False

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
        This method sets the :class:`gensim.models.word2vec.Word2Vec` model to be used in the syntactic model.

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

        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if not os.path.exists(self.corpus_filepath):
            raise FileNotFoundError(msg.CORPUS_NOT_GENERATED)
        
        if epochs is None:
            epochs = self.w2v_model.epochs

        sentences = LineSentence(self.corpus_filepath)
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
        self.generate_corpus(save=self._save_corpus, with_annotations=self._with_annotations)
        self.update_w2v_model = True
        

    def from_pretrained(self, model):
        if not isinstance(model, str):
            raise TypeError("Parameter model must be a string pointing to the Word2Vec model file.")

        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")
        
        self._is_pretrained = True
        if not isinstance(model, str):
            raise TypeError

        self.w2v_model = Word2Vec.load(model)
    

    
