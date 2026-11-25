HiT (Hierarchy Transformer)
===========================

HiT [he2024hit]_ turns a pre-trained sentence encoder into an encoder of a class hierarchy.
The catalogue key is ``hit``; its card in the :doc:`method catalogue <catalogue>` lists the
(few) claims with their sources. The model is :class:`mowl.models.hit.HiT`. It is a combination of parts (the function
``hit`` in :mod:`mowl.representation.library.hierarchy_methods`): the encoder with the
Poincaré ball it is read in is the class model, ``distance and depth`` is the measure, and the
triplet losses are those of the geometry. The module is :class:`mowl.nn.lm.hit.HiTModule`
and the Poincaré-ball operations are in :mod:`mowl.representation.library.poincare_ball`; the earlier direct
implementation is kept in :mod:`mowl.baselines.models.hit` as the reference the combination is
tested against.
HiT belongs to the ``language_model`` family: concepts are not regions but the encodings
of their labels.

Geometry
--------

A class :math:`A` is embedded as :math:`x_A = \mathrm{enc}(V(A))`, the mean-pooled output of
a transformer applied to its label :math:`V(A)` (``rdfs:label``, or the IRI fragment; see
:mod:`mowl.text.verbalise`). The outputs are read as points of the Poincaré ball of
curvature :math:`-c` with :math:`c = 1/d`, the ball of radius :math:`\sqrt{d}` that contains
the hypercube :math:`[-1, 1]^d`. With the hyperbolic distance :math:`d_c` and the hyperbolic
norm :math:`\lVert x\rVert_c = d_c(x, 0)`, a training triplet :math:`(e, e^+, e^-)` of a
class, a parent and a negative parent costs

.. math::

   \max(d_c(e, e^+) - d_c(e, e^-) + \alpha, 0) + \max(\lVert e^+\rVert_c - \lVert e\rVert_c
   + \beta, 0),

the hyperbolic clustering and centripetal losses (Eqs. 3-5): children are closer to their
parents than to other classes, and parents are closer to the origin. A subsumption is
scored by :math:`s(C \sqsubseteq D) = -(d_c(x_C, x_D) + \lambda(\lVert x_D\rVert_c -
\lVert x_C\rVert_c))` (Eq. 6), with :math:`\lambda` chosen on validation data.

HiT covers atomic subsumptions only. It has no set-theoretic semantics and the paper
proves no soundness or faithfulness result. Because the encoder reads text, it embeds
classes that were not seen in training, or classes of another ontology.

Training
--------

The positives are the atomic subsumptions of the training ontology (``taxonomy="asserted"``)
or the direct subsumptions found by ELK (``taxonomy="direct"``, as the paper builds
hierarchies from OWL ontologies). Each positive gets ``num_negatives`` (10) negative parents:
random classes that are not superclasses under the closed-world assumption, or siblings
first (``hard_negatives=True``). The defaults follow the paper: :math:`\alpha = 5`,
:math:`\beta = 0.1`, AdamW with learning rate :math:`10^{-5}`, 500 warm-up steps, batch size
256. After each epoch the model ranks the superclasses of the validation subsumptions, picks
:math:`\lambda` and keeps the best epoch by MRR. The paper selects by F1 with a tuned
threshold instead; see the card notes.

The encoder is a ``sentence-transformers`` model (an optional dependency, ``pip install
sentence-transformers``), loaded from its name with mean pooling and without the
normalisation layer, or any :class:`torch.nn.Module` that maps a list of strings to a
tensor. :class:`~mowl.representation.library.text_encoders.BagOfWordsEncoder` is a small stand-in for tests and
examples; it has none of the knowledge of a pre-trained language model.

Example
-------

The example uses the stand-in encoder, so it runs without the optional dependency. The
family ontology has no labels; ``label_fallback="camel_case"`` turns ``hasChild`` into
"has child".

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models.hit import HiT
   from mowl.representation.library.text_encoders import BagOfWordsEncoder

   dataset = FamilyDataset()
   model = HiT(dataset, encoder=BagOfWordsEncoder(dim=16), label_fallback="camel_case",
               batch_size=16, learning_rate=0.01, warmup_steps=0, num_negatives=2)
   model.train(epochs=2)
   vectors = model.class_embeddings          # {IRI: vector in the Poincaré ball}
   assert len(vectors) == len(dataset.classes)

With ``sentence-transformers`` installed, the encoder is named instead (the block is
skipped when the package is missing; it downloads the model):

.. testcode::
   :skipif: __import__("importlib").util.find_spec("sentence_transformers") is None

   model = HiT(dataset, encoder="sentence-transformers/all-MiniLM-L6-v2",
               label_fallback="camel_case", batch_size=16)
   model.train(epochs=1)

Benchmarks
----------

The HiT paper evaluates on WordNet, SNOMED CT and transfer to Schema.org, FoodOn and DOID
with threshold classification, which mOWL does not ship; its training design is the named
regime ``hit-code-training``. The OnT paper [yang2025ont]_
reports HiT on GALEN, GO and ANATOMY under its own normalisation (``ont-*`` benchmarks);
those numbers were obtained with the OnT training pipeline (``ont-code-training``), and
the ``hit`` recipes on the ``ont-*`` benchmarks follow it. See :doc:`ont`.
