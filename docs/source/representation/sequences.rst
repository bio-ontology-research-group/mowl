Methods that learn from sequences
=================================

DL2Vec, OWL2Vec*, Onto2Vec, OPA2Vec and random walks on the class hierarchy have no regions
and no relation model. They turn the ontology into sequences of tokens and train a word
embedding on them. Every class, individual and relation is then a point, and the only thing
the model defines is how similar two names are.

These methods are made of other parts than the region methods, and each part can be
exchanged on its own (:mod:`mowl.representation.sequence`):

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Part
     - Code
     - What it decides
   * - View
     - ``ProjectionView``, ``AxiomCorpus``, ``AnnotationCorpus``
     - What is read from the ontology: a graph made by a projector of
       :mod:`mowl.projection` (taxonomy, DL2Vec, OWL2Vec*, ...) or sentences (axioms,
       annotations). An ``Entailment`` adds the axioms a reasoner entails first.
   * - Sequences
     - ``Walks``, ``Sentences``, ``Mixed``
     - Random walks of a walker of :mod:`mowl.walking` (DeepWalk, node2vec) on a graph view,
       the sentences of a corpus view, or several of these in one corpus.
   * - Encoder
     - ``Word2VecEncoder``, ``FastTextEncoder``
     - What turns the sequences into one vector per token: Word2Vec (skip-gram or CBOW, with
       all its options) or fastText.
   * - Reading
     - ``TokenVectors``
     - Which vector an entity gets: the vector of the token that is its IRI.
   * - Measure
     - ``DotSigmoid``, ``Cosine``
     - How similar two entities are; this is the geometry's "inside", and it is symmetric.
   * - Scorer
     - ``PairClassifier``, ``SimilarityScorer``
     - The score of a candidate axiom for the ranking of an evaluation regime: a classifier
       on the two vectors, trained on the training axioms for each relation, or the
       similarity of the two entities.

The parts are in :mod:`mowl.representation.library.walks_views` and
:mod:`mowl.representation.library.walks_encoders`. The published methods are combinations
of them (:mod:`mowl.representation.library.walks_methods`), and
:class:`~mowl.models.ComposedSequenceModel` trains any combination.

Build a published method by name
--------------------------------

.. testcode::

   import logging
   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ComposedSequenceModel

   logging.disable(logging.CRITICAL)
   dataset = FamilyDataset()

   model = ComposedSequenceModel(dataset, method="onto2vec", embed_dim=16, workers=1)
   model.train(epochs=5)
   parts = model.module.parts
   print(parts["corpus"], "/", parts["encoder"], "/", parts["scorer"])

.. testoutput::

   axioms as sentences / word2vec skip-gram / classifier on pairs of vectors

The classes users have always used are the same combinations:
``RandomWalkPlusW2VModel`` with a projector and a walker is the walk method of that
projector, and ``SyntacticPlusW2VModel`` with ``generate_corpus`` is Onto2Vec (OPA2Vec with
``with_annotations=True``). Their ``module`` is the geometry with these parts.

Swap a part
-----------

A part is swapped by an argument of the method or by passing a part. Here DL2Vec walks with
node2vec instead of DeepWalk, encodes with CBOW instead of skip-gram, and the sentences of the
axioms are added after the walks:

.. testcode::

   model = ComposedSequenceModel(dataset, method="dl2vec", walker="node2vec", sg=0,
                                 corpus=("axioms",), embed_dim=16, num_walks=2,
                                 walk_length=5, workers=1)
   parts = model.module.parts
   print(parts["sequences"], "|", parts["walker"], "|", parts["encoder"])

.. testoutput::

   walks + sentences | node2vec walks | word2vec CBOW

Parts can also be put together without a named method:

.. testcode::

   from mowl.representation.library.walks_views import (projection_view, make_walker,
                                                        Walks, Mixed, Sentences,
                                                        AnnotationCorpus)
   from mowl.representation.library.walks_encoders import FastTextEncoder, SimilarityScorer

   source = Mixed(Walks(projection_view("taxonomy"), make_walker("deepwalk", 2, 5)),
                  Sentences(AnnotationCorpus()))
   model = ComposedSequenceModel(dataset, source=source,
                                 encoder=FastTextEncoder(vector_size=16, min_count=1,
                                                         workers=1),
                                 scorer=SimilarityScorer())
   print(model.module.parts["encoder"])

.. testoutput::

   fastText CBOW

Evaluate
--------

A sequence model is evaluated by the same regimes as every other model. The regimes ask the
model for the costs of candidate axioms, and the scorer of the geometry gives them
(``scoring="auto"``). The scorer can also be passed to a regime directly, since it has the
signature of a scorer of :mod:`mowl.evaluation`:

.. testcode::

   from mowl.evaluation.filtered_ranking import common_ranking
   from mowl.representation.library.walks_encoders import SimilarityScorer

   from mowl.datasets import Dataset

   # the training axioms are the test axioms here, only to show the call
   both = Dataset(dataset.ontology, testing=dataset.ontology)
   model = ComposedSequenceModel(both, method="taxonomy_walks", embed_dim=16,
                                 num_walks=2, walk_length=5, workers=1)
   model.train(epochs=5)
   regime = common_ranking("gci0", filter_by=("train",), scoring=SimilarityScorer())
   result = regime.evaluate(model, both)
   print(sorted(result["all"])[:3])

.. testoutput::
   :options: +ELLIPSIS

   ['f_auc', 'f_hits@1', 'f_hits@10']

Train with a regime
-------------------

With a training regime the encoder is trained one pass at a time, and the regime is asked
before and after each pass. ``RefreshSequences`` draws new walks before every pass, so the
encoder does not see the same walks again:

.. testcode::

   from mowl.training.sequences import RefreshSequences

   model = ComposedSequenceModel(dataset, method="dl2vec", embed_dim=16, num_walks=2,
                                 walk_length=5, workers=1)
   model.training_regime = RefreshSequences(every=1)
   model.train(epochs=2)
   print(model.training_regime.refreshed)

.. testoutput::

   [1]

Defaults
--------

The parts default to the correct behaviour (see :doc:`defaults`). Four behaviours of the
earlier direct implementation are options:

* A new walk replaces the walks of an earlier one. The walker of DeepWalk appended to its
  file, so a second walk used to train on the old and the new walks together; the walkers now
  empty their file first (``DeepWalk(append=True)`` keeps the appending).
* A Word2Vec model that has a vocabulary keeps its vectors when it is trained again, so
  ``from_pretrained`` followed by ``train`` goes on from the loaded vectors
  (``restart=True`` starts anew from the seed on every ``train``, as before).
* The start vectors are seeded with a hash that is the same in every Python process, so the
  same ``seed`` with one worker gives the same vectors in every run (``hashfxn=hash`` uses
  Python's hash, which changes between processes unless ``PYTHONHASHSEED`` is set).
* An annotation sentence names its class by the IRI, the same token as in the axiom
  sentences and the walks (``AnnotationCorpus(entity_form="bracketed iri")`` writes
  ``<IRI>``, a token that no axiom sentence contains, as the saved corpus of
  :mod:`mowl.corpus` did before it got the same ``entity_form`` option, with ``"iri"`` as its
  default).

The comparison with the direct implementation (``tests/baselines/pairs_walks.py``) selects
these options and checks that both give the same sequences, vocabulary, vectors and scores.
