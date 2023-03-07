from mowl.base_models.graph_model import RandomWalkModel
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import mowl.error.messages as msg

class RandomWalkPlusW2VModel(RandomWalkModel):

    def __init__(self, *args, **kwargs):
        super(RandomWalkPlusW2VModel, self).__init__(*args, **kwargs)

        self._edges = None
        self.w2v_model = None
        self.update_w2v_model = False

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
        self.w2v_model = Word2Vec(*args, **kwargs)

    def train(self, epochs=None):
        if self.projector is None:
            raise AttributeError(msg.GRAPH_MODEL_PROJECTOR_NOT_SET)
        if self.walker is None:
            raise AttributeError(msg.RANDOM_WALK_MODEL_WALKER_NOT_SET)
        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if epochs is None:
            epochs = self.w2v_model.epochs

        if self._edges is None:
            self._edges = self.projector.project(self.dataset.ontology)
            self.walker.walk(self._edges)
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
        
    #TODO: implement from pretrained
    def from_pretrained(self, model):
        self.is_pretrained = True
        if not isintance(model, str):
            raise TypeError

        self.w2v_model = Word2Vec.load(model)
    #    set_w2v_model(model)

    
