Embedding with KGE methods
===========================

.. |edge| replace:: :class:`Edge <mowl.projection.edge.Edge>`

PyKEEN integration
-----------------------

Generating graphs from ontologies opens a wide range of possibilities on Knowledge Graph Embeddings. `PyKEEN <https://pykeen.readthedocs.io/en/stable/index.html>`_ is a Python package for reproducible, facile knowledge graph embeddings. mOWL provides some functionalities to ease the integration with PyKEEN methods that are subclasses of :class:`pykeen.models.base.EntityRelationEmbeddingModel` or :class:`pykeen.models.nbase.ERModel`. After :doc:`generating a graph from an ontology <projection>`, the output is a list of |edge|. It is possible to transform this list to a PyKEEN :class:`pykeen.triples.TriplesFactory` class:

.. testcode:: [pykeen]

   from mowl.projection.edge import Edge
   from mowl.datasets.builtin import PPIYeastSlimDataset
   from mowl.projection import TaxonomyProjector

   ds = PPIYeastSlimDataset()
   proj = TaxonomyProjector(True)

   edges = proj.project(ds.ontology)
   
   #edges = [Edge("node1", "rel1", "node3"), Edge("node5", "rel2", "node1"), Edge("node2", "rel1", "node1")] # example of edges
   triples_factory = Edge.as_pykeen(edges, create_inverse_triples = True)

.. note::
   The ``create_inverse_triples`` parameter belongs to PyKEEN triples factory method.

Now, this triples factory can be used to call a PyKEEN model:

.. testcode:: [pykeen]

   from pykeen.models import TransE
   pk_model = TransE(triples_factory=triples_factory, embedding_dim = 50, random_seed=42)

   
At this point, it is possible to continue in either in PyKEEN or mOWL environments. mOWL :class:`mowl.kge.model.KGEModel` wraps the :class:`pykeen.training.SLCWATrainingLoop` construction:

.. testcode:: [pykeen]

   from mowl.kge import KGEModel

   model = KGEModel(triples_factory, pk_model, epochs = 10, batch_size = 32)
   model.train()
   ent_embs = model.class_embeddings_dict
   rel_embs = model.object_property_embeddings_dict

.. attention::
   PyKEEN might generate more than one embedding vector per entity. However, in mOWL wrapping class only the primary embedding vector is returned.



  
Generating embeddings using a mOWL model
-------------------------------------------------

Although the embedding generations can be done step by step, we also provide a class that performs all the steps internally:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import GraphPlusPyKEENModel
   from mowl.projection import DL2VecProjector
   from pykeen.models import TransE
   import torch as th
   
   model = GraphPlusPyKEENModel(FamilyDataset())
   model.set_projector(DL2VecProjector())
   model.set_kge_method(TransE, random_seed=42)
   model.optimizer = th.optim.Adam
   model.lr = 0.001
   model.batch_size = 32
   model.train(epochs = 2)

   # Get embeddings

   class_embs = model.class_embeddings
   role_embs = model.object_property_embeddings
   ind_embs = model.individual_embeddings

The parts of the model
-------------------------------------------------

The model is a combination of parts: the projection, the entity and relation representations of the PyKEEN model with its interaction function, the loss of the PyKEEN model as the objective, and the negative sampler. ``model.module`` is the :class:`~mowl.representation.Geometry` that names them, and the model trains on the triples of the graph with the loop of :class:`~mowl.models.graph_kge.triple_model.TripleGraphModel` (``num_negs_per_pos`` corrupted triples per positive, the optimiser, learning rate and batch size set above). As in the released class (mOWL 2.1.0), ``train`` takes the steps of PyKEEN's sLCWA loop: every call draws the parameters again (after PyKEEN's memory probe, which splits a batch that does not fit on the GPU into sub-batches) and trains on PyKEEN's batches and negatives. ``model.reinitialise = False`` continues from the current parameters instead (a second ``train`` goes on where the first stopped, pretrained parameters are kept), and ``model.batching = "mowl"`` uses mOWL's batches and sampler, which a training regime needs to change the negatives, the loss or the regularisation.

.. testcode::

   parts = model.module.parts
   print(parts["measure"], "|", parts["negative sampler"], "|", parts["projection"])

.. testoutput::

   triple score | uniform corruption | DL2VecProjector

:class:`~mowl.models.GraphKGEModel` builds the same combination from the registered methods of mOWL (``"boxe"``, ``"expressive"``, ``"reshuffle"``, ``"graph_kge"``, ``"gda_projection"``) with every part swappable, and :class:`~mowl.models.BoxE`, :class:`~mowl.models.ExpressivE` and :class:`~mowl.models.Reshuffle` are its named combinations.
