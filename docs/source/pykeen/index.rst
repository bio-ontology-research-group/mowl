PyKEEN integration
=======================


Generating graphs from ontologies opens a wide range of possibilities on Knowledge Graph Embeddings. `PyKEEN <https://pykeen.readthedocs.io/en/stable/index.html>`_ is a Python package for reproducible, facile knowledge graph embeddings. mOWL provides some functionalities to ease the integration with PyKEEN methods that are subclasses of :class:`pykeen.models.base.EntityRelationEmbeddingModel` or :class:`pykeen.models.nbase.ERModel`. After :doc:`generating a graph from an ontology </projection/index>`, the output is a list of :class:`Edge <moowl.projection.edge.Edge>`. It is possible to transform this list to a PyKEEN :class:`pykeen.triples.TriplesFactory` class:

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
