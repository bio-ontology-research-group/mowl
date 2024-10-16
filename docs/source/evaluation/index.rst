Evaluating the embeddings
===============================

.. |ppidataset| replace:: :class:`PPIYeastDataset <mowl.datasets.builin.PPIYeastDataset>`
.. |ppievaluator| replace:: :class:`PPIEvaluator <mowl.evaluation.PPIEvaluator>`
			  
The evaluation of models is specific to the task, which is given by the dataset. For example, when using the |ppidataset|, we will use the |ppievaluator|. For example, a typical pipeline to train a model would be:

.. testcode::
   
   from mowl.datasets.builtin import PPIYeastSlimDataset
   from mowl.models import SyntacticPlusW2VModel

   model = SyntacticPlusW2VModel(dataset, corpus_filepath="test")
   model.set_w2v_model(min_count=1)
   model.generate_corpus(save=True, with_annotations=True)
   model.train()

To evaluate, we first need to assign the evaluator:
   
.. testcode::
   
   from mowl.evaluation import PPIEvaluator
   model.set_evaluator(PPIEvaluator)
   model.evaluate()


What characterizes each evaluator class are two things:

* The type of entities involved in the evaluation
* The type of axiom to be evaluated

In the case of the |ppievaluator|, the entities involved in the evaluation are only those ones representing entities, we do not consider other entities present in the ontologies such as GO functions.

For the |ppievaluator|, the axioms to be evaluated is :math:`p_i \sqsubseteq \exists interacts\_wih. p_j`, where :math:`p_i` and :math:`p_j` are proteins.

Every dataset has the attribute `evaluation_classes`, which is a 2-tuple of objects :class:`mowl.datasets.OWLClasses`. 
