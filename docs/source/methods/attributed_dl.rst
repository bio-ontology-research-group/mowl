Convex models of (temporally) attributed ontologies
===================================================

Attributed description logics annotate axioms and assertions with *specifiers*: sets of
attribute-value pairs such as ``[source: s, time: 3]``, or open specifiers ``⌊source: s⌋``
("at least these pairs"). Bourgaux, Ozaki and Pan ([bourgaux2021]_) show that every satisfiable
ontology of :math:`DL\text{-}Lite^{\mathcal{H},@}_{horn}` without negative role inclusions
has a convex geometric model: ground the ontology (every annotated name ``E@S`` becomes a
fresh name ``E_S``, and ``E_S ⊑ E_T`` is added when ``S ⇒ T``), and apply the construction
of Gutiérrez-Basulto and Schockaert to the resulting ordinary ontology (Theorem 6). With
temporal attributes this holds if role specifiers are atemporal and ``before``, ``after``
and ``between`` do not occur (Theorem 12); without these conditions it fails (Example 10).
The paper has no learning procedure. mOWL implements the construction (catalogue key
``attributed_dl``, status *constructive*) in :mod:`mowl.geometric.attributed`.

How annotations are supplied
----------------------------

Annotations are kept apart from the geometry. They are either Python objects
(:class:`~mowl.geometric.attributed.AttributedOntology` with ``ConceptFact``,
``RoleFact``, ``ConceptInc``, ``RoleInc`` and :class:`~mowl.geometric.attributed.Specifier`),
or *OWL axiom annotations*: :func:`~mowl.geometric.attributed.from_owl_ontology` reads
every annotation whose property is in the namespace ``http://mowl.borg/attribute#`` as an
attribute (``.../attribute#time`` with the literal ``"3"``, ``.../attribute#during`` with
``"1,5"``). Annotations on an assertion are its specifier; for a ``SubClassOf`` or
``SubObjectPropertyOf`` axiom they are the same closed specifier on all names of the axiom
(``A@S ⊑ B@S``, an adaptation; use the Python API for different specifiers on both sides).
Axioms without annotations are ordinary axioms, and axioms outside the supported forms are
returned in a list of skipped axioms.

Supported
---------

* Ground axioms: ``A(a)@S``, ``R(a, b)@S``, ``B1 ⊓ ... ⊓ Bk ⊑ C`` with ``Bi`` either
  ``A@S`` or ``∃R@S`` and ``C`` either of these or ``⊥``, and ``R@S ⊑ Q@T``. Expressivity
  :math:`\mathcal{ELH}_\bot` with ABox, no inverse roles.
* Temporal attributes ``time``, ``during``, ``since`` and ``until`` on concepts in closed
  specifiers. The grounding of these is ours: the entailments of Definition 7 of the paper
  between them are added as Horn axioms (``during`` and ``until`` are conjunctions of
  ``time`` points; ``since:x`` implies the later time points that occur in the ontology).
* Not supported: set variables and object variables (only ground ontologies), inverse roles,
  negative role inclusions, ``before``/``after``/``between``, temporal specifiers on roles
  (these raise :class:`~mowl.geometric.errors.UnsupportedAxiom`).

The model is exact: ``is_model()`` checks every original axiom on the geometry, and an
annotated assertion holds geometrically iff the grounded ontology entails it.

Example
-------

.. testcode::

   from mowl.geometric.attributed import (
       AttributedConvexModel, AttributedOntology, Basic, ConceptFact, ConceptInc, Specifier)

   s, t = Specifier.closed(source="s"), Specifier.closed(source="t")
   onto = AttributedOntology([
       ConceptFact("A", "a", s), ConceptFact("A", "b", t),
       ConceptInc((Basic("A", s),), Basic("B", s)),       # only under annotation [source: s]
   ])
   model = AttributedConvexModel(onto)
   print(model.is_model())
   print(model.holds_concept("a", "B", s), model.holds_concept("b", "B", s))
   print(model.holds_concept("a", "A"))                    # unannotated: any annotation set

.. testoutput::

   True
   True False
   True
