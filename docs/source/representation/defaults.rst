Defaults: correct, or as published
==================================

Publications and their code sometimes do something that is not what the text says: a sign that
is not constrained, a loss that can go below zero, a role that is paired with the wrong
element. mOWL keeps one rule for such quirks. A part defaults to the correct reading, and a
named method may keep its authors' reading as an option. In detail:

* **A part defaults to the correct behaviour.** A part is meant to be combined with other parts and
  used on your own ontologies, so it must not bring a quirk of one paper into a combination
  that has nothing to do with that paper.
* **A named method may keep its authors' behaviour.** ``ComposedEL(dataset, method="boxel")``
  and the benchmark recipes of a publication set the options that give the published numbers.
* **The quirk is an option, never lost.** The option has a name, the docstring of the part says
  what it does, and :doc:`known_deviations` lists what a method does that way.
* **A defect in an existing module is fixed there.** The correct behaviour becomes the default,
  the old behaviour stays available as an option, and the published recipes select it. Every
  such change is in the changelog.

Two examples
------------

**A role that turns a box inside out.** In BoxEL a role scales the boxes of its fillers. The
authors' code lets the scale have any sign. A negative scale then maps a box to an *inverted*
box, one whose half-width is negative, and everything computed from it is meaningless. The
part ``Affine`` takes the absolute value by default, and the box stays a box. The signed scale
of the authors stays available as ``scale="signed"``, and the method ``boxel`` takes the same
option:

.. testcode::

   import inspect
   import torch as th
   from mowl.representation import Box, METHODS
   from mowl.representation.library.boxrel_relations import Affine

   box = Box(th.zeros(1, 2), th.ones(1, 2))      # the square [-1, 1] x [-1, 1]
   relation = Affine(1, 2)                        # one relation, two dimensions
   relation.scales.weight.data.fill_(-1.0)        # a negative scale
   ids = th.tensor([0])

   print("part, default:   ", relation.image(ids, box).offset.round().tolist())
   relation.scale = "signed"                      # the authors' reading
   print("part, as BoxEL:  ", relation.image(ids, box).offset.round().tolist())
   print("method 'boxel':  scale =", inspect.signature(METHODS["boxel"]).parameters["scale"].default)

.. testoutput::

   part, default:    [[1.0, 1.0]]
   part, as BoxEL:   [[-1.0, -1.0]]
   method 'boxel':  scale = positive

The module of BoxEL itself was changed in the same way. ``BoxEL(..., positive_scale=True)`` is
the default, and ``positive_scale=False`` restores the signed scale (BoxSEL follows).
The method ``boxel`` reads ``scale="positive"`` by default and ``scale="signed"`` for the
released code.

**A loss that goes below zero.** Shadow cones score a pair by the signed distance to the cone
of the parent. The authors' loss for a pair that does *not* belong together is that raw
number, so it is smallest, and most negative, deep inside the cone, which is the wrong way
round for a loss. The part scores a wrong pair with a hinge, :math:`\max(0, \gamma - E)`, and
so does the method ``shadow_cones``; the raw energy is the option ``negative_form="energy"``:

.. testcode::

   from mowl.representation.library.hierarchy_measures import ShadowCone

   part = ShadowCone()
   method = METHODS["shadow_cones"](4).measure
   print("part:   signed =", part.signed, " negative =", part.negative_form)
   print("method: signed =", method.signed, " negative =", method.negative_form)

.. testoutput::

   part:   signed = False  negative = hinge
   method: signed = True  negative = hinge

The module ``ShadowConesModule`` behaves in the same way: ``negative="hinge"`` is its default
and ``negative="energy"`` gives the old loss. For the numbers of the paper (its Equation 1) the
two are identical, so training with the corrected default gives the same model.

Where the rule shows up
-----------------------

* A cone's aperture is kept in :math:`[0, \pi]` by default (``aperture="clip"``). The AConE
  method and model let it wrap to negative values, as the authors' code does (``"wrap"``).
* The negation of a cone uses the absolute aperture by default (``signed_negation=False``,
  ConE and the parts). The method ``acone`` uses :math:`\pi - \text{aperture}` as published
  (``signed_negation=True``). The rotation of AConE maps the preimage of a cone the way it maps
  the image (``aperture_change="signed"`` is the authors' reading).
* A box intersection is exact: two boxes that are apart give the empty box
  (``Box.intersect(signed=False)`` gives the earlier box with a positive width).
* A loss for a missing negative is an error, not the positive loss
  (``strict_negatives=False`` gives the old behaviour).
* A loss for a wrong pair is measured in the same sense as the margin (a slack stays a
  slack). ELEmbeddings, EmEL++ and RegD-ELEm use the strict reading of their authors.
* FALCON's quantifiers look at all pairs of entities. The first mOWL version paired every
  entity with itself (``role_pairs="self"`` restores that), and it learned ``owl:Nothing``
  (``nothing="learned"``).
* Al-cone gives ``owl:Thing`` and ``owl:Nothing`` a constant logit on the scale of the learned
  ones; ``constant_logit=1e4`` restores the earlier, much larger constant.

What changed in default behaviour
---------------------------------

This is for users of earlier versions (v2.1.0 and before). Each line names the method, what is
different now, and how to get the old behaviour. Models that did not exist in v2.1.0 have no
earlier behaviour and are not listed. The changelog has every change.

* **All models with a per-axiom loss** (``ELEmbeddings``, ``ELBE``, ``BoxEL``, ``Box2EL``,
  ``TransBox`` and the others built on ``ELModule``): every loss has shape ``(B, 1)``;
  ``module.per_axiom_column = False`` restores the flat ``(B,)``. Asking for a negative loss of
  a form that has none raises ``NotImplementedError``; ``strict_negatives = False`` restores
  the silent positive loss.
* **ELEmbeddings** (and ``RegDELEm``): the loss of :math:`C \sqcap D \sqsubseteq E` has the
  fourth term of the published loss; ``gci1_radius_term=False`` restores the old loss.
* **ELBE**: the loss of :math:`C \sqcap D \sqsubseteq E` is one value per sample (it was an
  ``n x n`` matrix that mixed samples). This was a defect, there is no option for the old
  numbers. The bottom losses have shape ``(B, 1)``.
* **BoxEL** (and ``BoxSEL``): the role scale is :math:`|s| + \epsilon`, so no role inverts a
  box (``positive_scale=False``); the role chain loss composes the maps in order
  (``role_chain_composition="as_released"``); the role inclusion and role chain losses can be
  trained (``load_role_axioms=True``); the loss of :math:`\exists r.C \sqsubseteq D` was fixed
  (a defect, no option).
* **Box2EL** (``BoxSquaredEL``): losses have shape ``(B, 1)``; the role inclusion and role
  chain losses of the paper can be trained (``load_role_axioms=True``). Nothing else changed.
* **TransBox**: a negative loss of a form without one raises (``strict_negatives=False``); with
  ``formulation="reference"`` the bottom forms follow the authors' code
  (``reference_bottom_losses="mowl"``).
* **FALCON**: the quantifiers look at all pairs of entities (``role_pairs="self"``), and
  ``owl:Nothing`` is the empty set (``nothing="learned"``). ALC models now train on a GPU.
* **Row order of the datasets** (``ELDataset``, ``ALCDataset``) and the edge order of the
  ``DL2VecProjector``, ``OWL2VecStarProjector`` and ``GDAProjector``: the ontology reader and
  the normaliser yield axioms in the order of a Java hash set, which differs between two builds
  in one process, and negatives are drawn per row, so equal seeds gave unequal runs. The rows of
  every form are now in a documented order: sorted by their integer indices and then permuted by
  a generator seeded with ``shuffle_seed`` (default 0; ``shuffle=False`` keeps them sorted;
  ``shuffle_seed=None`` uses the global :mod:`random` module as before). ALC patterns and rows,
  the edges of the three projectors and the lines of the corpora are sorted. The index
  dictionaries were already sorted.
* **Corpora** (:mod:`mowl.corpus`): ``extract_and_save_annotation_corpus`` writes a class as
  its IRI, like the axiom corpus and the list variant (it wrote ``<IRI>``, so annotation
  sentences never reached the class vectors); ``entity_form="bracketed iri"`` gives the old
  form. ``SyntacticPlusW2VModel.generate_corpus`` has the same option
  (``annotation_entity_form``), and the baseline OPA2Vec selects the old form.
* **Walkers** (:mod:`mowl.walking`): the output file is emptied before every walk
  (``DeepWalk(append=True)`` keeps the old appending, e.g. to extend the walks of a graph that
  grew). DeepWalk's Java code appends to its file, and Node2Vec has no seed in the Scala
  source: both need a rebuild of the jar to be fixed there, so they are handled in Python
  (``walker.seeded`` is ``False`` for node2vec).
* **FaithEL**: the part ``AnchoredObjective`` has a bounded negative term by default
  (``negative_term="margin"``); the class ``FaithEL`` keeps the unbounded term of the authors'
  code (``negative_term="unbounded"`` is its default).
* **FALCON**: the part ``FuzzyObjective`` reads a role assertion in the direction of
  :math:`\exists r.C` by default (``abox_head="subject"``); the named method keeps the
  direction of the code (``"object"``), see :doc:`known_deviations`.
* **Graph and syntactic methods** (``GraphPlusPyKEENModel``, ``RandomWalkPlusW2VModel``,
  ``SyntacticPlusW2VModel``): ``RandomWalkPlusW2VModel`` and ``SyntacticPlusW2VModel`` are
  combinations of parts (:doc:`sequences`). A second walk replaces the first instead of being
  appended, training goes on from the vectors there are (``restart=True`` starts anew),
  start vectors are seeded the same way in every process (``hashfxn=hash`` as before), and
  annotation sentences name a class by its IRI. ``GraphPlusPyKEENModel.axiom_costs`` now
  scores class and role assertions too, not only :math:`C \sqsubseteq D` and
  :math:`C \sqsubseteq \exists r.D`. Datasets are cached under ``$XDG_CACHE_HOME/mowl``
  (override with ``MOWL_DATA_ROOT``).
