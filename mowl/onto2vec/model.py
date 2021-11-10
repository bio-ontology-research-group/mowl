import os
from mowl.model import Model
from jpype.types import *

from org.semanticweb.owlapi.manchestersyntax.renderer import ManchesterOWLSyntaxOWLObjectRendererImpl
from org.mowl.Onto2Vec import Onto2VecShortFormProvider
from org.semanticweb.owlapi.model import AxiomType
import gensim
import logging

class CorpusGenerator(object):

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath) as f:
            for line in f:
                yield line.strip().split(' ')

class Onto2Vec(Model):

    def __init__(self, dataset, w2v_params={}):
        super().__init__(dataset)
        self.axioms_filepath = os.path.join(
            dataset.data_root, dataset.dataset_name, 'axioms.o2v')
        self.w2v_params = w2v_params
        self.model_filepath = os.path.join(
            dataset.data_root, dataset.dataset_name, 'w2v.model')
        self.w2v_model = None
        
    def _create_axioms_corpus(self):
        logging.info("Generating axioms corpus")
        renderer = ManchesterOWLSyntaxOWLObjectRendererImpl()
        shortFormProvider = Onto2VecShortFormProvider()
        renderer.setShortFormProvider(shortFormProvider)
        with open(self.axioms_filepath, 'w') as f:
            for owl_class in self.dataset.ontology.getClassesInSignature():
                axioms = self.dataset.ontology.getAxioms(owl_class)
                for axiom in axioms:
                    rax = renderer.render(axiom)
                    rax = rax.replaceAll(JString("[\\r\\n|\\r|\\n()]"), JString(""))
                    f.write(f'{rax}\n')

    def train(self):
        if not os.path.exists(self.axioms_filepath):
            self.dataset.infer_axioms()
            self._create_axioms_corpus()

        sentences = CorpusGenerator(self.axioms_filepath)
        self.w2v_model = gensim.models.Word2Vec(
            sentences=sentences, **self.w2v_params)
        self.w2v_model.save(self.model_filepath)


    def evaluate_ppi(self):
        if not os.path.exists(self.model_filepath):
            self.train()
        if not self.w2v_model:
            self.w2v_model = gensim.models.Word2Vec.load(
                self.model_filepath)
        classes = self.dataset.get_evaluation_classes()
        classes = [str(cls.toString()) for cls in classes]
        for axiom in self.dataset.testing.getAxioms():
            if axiom.getAxiomType() != AxiomType.SUBCLASS_OF:
                continue
            cls1 = axiom.getSubClass()
            cls2 = axiom.getSuperClass().getFiller()
            cls1, cls2 = str(cls1), str(cls2)
            print(cls1, cls2)
        
    
