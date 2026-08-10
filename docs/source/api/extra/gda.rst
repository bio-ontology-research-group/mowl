

In addition to all the OWL2Vec\* rules, the following rule is applied. It fires
only when the OWL2Vec\* rules produce no triple, which is exactly when the
filler of the existential restriction is not an atomic class.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Axiom of condition 1
     - Condition 2
     - Projected triple(s)
   * - :math:`A \sqsubseteq \exists r_{1} . (C_{1} \sqcap \ldots \sqcap \exists r_{2} . ( \ldots \exists r_{n} . B \ldots ))`
     - :math:`A` matches ``source_prefixes`` and :math:`B` matches ``target_prefixes``
     - :math:`\langle A, r_{1}\_\ldots\_r_{n}, B\rangle`

The relation of the projected triple is the concatenation of the local names of
the roles :math:`r_{1}, \ldots, r_{n}` traversed on the way to :math:`B`, joined
by ``_`` and minted under the ``http://mowl.borg/`` namespace. The walk descends
through intersections and nested existentials, and emits one triple per named
class it reaches that matches ``target_prefixes``.

.. warning::

   The composed relation name is not injective: two different role paths whose
   local names concatenate to the same string yield the same relation. The
   naming scheme is kept as published for reproducibility.
