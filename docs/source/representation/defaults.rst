Defaults: correct, or as published
==================================

Publications and their code sometimes do something that is not what the text says: a sign that
is not constrained, a loss that can go below zero, a role that is paired with the wrong
element. mOWL keeps one rule for such quirks:

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
  method lets it wrap to negative values, as the authors' code does (``"wrap"``).
* The negation of a cone uses the absolute aperture by default. The methods ``cone_qe`` and
  ``acone`` use :math:`\pi - \text{aperture}` as published (``signed=True``).
* A loss for a wrong pair is measured in the same sense as the margin (a slack stays a
  slack). ELEmbeddings, EmEL++ and RegD-ELEm use the strict reading of their authors.
* FALCON's quantifiers look at all pairs of entities. The first mOWL version paired every
  entity with itself (``role_pairs="self"`` restores that), and it learned ``owl:Nothing``
  (``nothing="learned"``).
* Al-cone gives ``owl:Thing`` and ``owl:Nothing`` a constant logit on the scale of the learned
  ones; ``constant_logit=1e4`` restores the earlier, much larger constant.

The changelog has the complete list of changed defaults.
