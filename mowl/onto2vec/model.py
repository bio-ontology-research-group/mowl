import os
from mowl.model import Model

from org.mowl import Onto2VecShortFormProvider
from org.semanticweb.owlapi.manchestersyntax.renderer import ManchesterOWLSyntaxOWLObjectRendererImpl

import gensim

class CorpusGenerator(object):

    def __init__(self, filename):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath) as f:
            for line in f:
                yield gensim.utils.simple_preprocess(line)

class Onto2Vec(Model):

    def __init__(self, dataset, w2v_params={}):
        super().__init__(dataset)
        self.axioms_filepath = os.path.join(
            dataset.data_root, dataset.dataset_name + '.o2v')
        self.w2v_params = w2v_params
        self.model_filepath = os.path.join(
            dataset.data_root, dataset.dataset_name + '.model')
        
    def _create_axioms_corpus(self):
        renderer = ManchesterOWLSyntaxOWLObjectRendererImpl()
        shortFormProvider = Onto2VecShortFormProvider()
        renderer.setShortFormProvider(shortFormProvider)
        with open(self.axioms_filepath, 'w') as f:
            for owl_class in self.dataset.ontology.getClassesInSignature():
                axioms = self.dataset.ontology.getAxioms(owl_class)
                for axiom in axioms:
                    rax = renderer.render(axiom)
                    rax = rax.replaceAll("[\\r\\n|\\r|\\n()]")
                    f.write(f'{rax}\n')

    def train(self):
        if not os.path.exists(self.axioms_filepath):
            self.dataset.infer_axioms()
            self._create_axioms_corpus()

        sentences = CorpusGenerator(self.axioms_filepath)
        self.w2v_model = gensim.models.Word2Vec(
            sentences=sentences, **self.w2v_params)
        self.w2v_model.save(self.model_filepath)
    
