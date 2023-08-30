mOWL model
============

There are different ways to generate embeddings for ontology entities. In mOWL, we have a prototypical class from which all models are created. This basic model is located at :class:`mowl.base_models.model.Model` and has the following methods:

Initialize and train
------------------------

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.base_models import Model

   dataset = FamilyDataset()
   model = Model(dataset, model_filepath="path_to_save_the_model")
   

Every model will receive a mOWL dataset. Furthermore, every model will have a ``model_filepath`` parameter where the model will be saved. In cases where models contain a Word2Vec instance, a :class:`gensim.models.word2vec.Word2Vec`  object will be saved in ``model_filepath``. In other cases, the object to be saved will be a :class:`torch.nn.Module`.

Some methods will need parameter initialization and after the initialization is done, the training can be done with:

.. code::

   model.train()


Getting embeddings
--------------------

After tranining, one way to get the learned embeddings is:

.. code::

   class_embs = model.class_embeddings
   role_embs = model.object_property_embeddings
   ind_embs = model.individual_embeddings

which correspond to the vectors for the ontology classes, roles and individuals, respectively.


Loading a pretrained model
------------------------------

With the following line:

.. code::

   model.from_pretrained("path_to_pretrained_model")

It is possible to load a previously trained model. Analogously to the initialization step, some methods will load a :class:`gensim.models.word2vec.Word2Vec` object and some methods will load a :class:`torch.nn.Module` object.


Adding axioms after training
-------------------------------------

In some cases, it might be necessary to augment the knowledge base by adding new axioms to the ontology. In that case, the model must be updated accordingly and introduce embedding vectors for the potential new entities. To add new axioms we can do:

.. code::

   axioms = # List of org.semanticweb.owlapi.model.OWLAxiom
   model.add_axioms(*axioms)



Every model in mOWL will contain these methods. However, depending on the type of model, other steps might need to be taken to initialize or train a model.

The types of models implemented in mOWL are:

* :doc:`Graph-based models <graphs/index>`
* :doc:`Syntactic models <corpus/index>`
* :doc:`Model-theoretic models <embedding_el/index>`
