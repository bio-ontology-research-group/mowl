MultiBoxEL (Aharoni, volume-based)
==================================

:class:`~mowl.models.mcmultibox.model.VolumeMultiBoxEL` implements the MultiBoxEL model of
[aharoni2024]_ (Chapter 7 of the thesis). The catalogue key is ``mcmultibox``. It is **not**
:class:`mowl.models.MultiBoxEL` (key ``multiboxel``), a different method that also uses unions of
boxes but trains Box\ :sup:`2`\ EL-style distance losses.

Geometry
--------

* A concept is a union of :math:`n` axis-parallel boxes in :math:`\mathbb{R}^d`, each stored as a
  centre and an offset (weights of shape :math:`|N_C| \times n \times 2d`).
* A role is a union of :math:`n` boxes in :math:`\mathbb{R}^{2d}`: the first :math:`d`
  coordinates are the domain, the last :math:`d` the codomain (:math:`|R| \times n \times 4d`).
* Individuals are single boxes (:math:`|O| \times 1 \times 2d`); the thesis defines no loss for
  them and mOWL does not train them.
* Volumes are measured inside :math:`S = [-b_o, b_o]^d` on a fixed set of quasi-random points (a
  scrambled Sobol sequence, deterministic given ``seed``). Membership of a point in a multibox is
  the soft function :math:`\tilde\chi_K(p) = \mathrm{relu}(\max_i m_i(p) - 0.5)` of the thesis,
  where :math:`m_i(p)` combines :math:`\sigma(M_i - p)` and :math:`\sigma(p - m_i)` over all
  coordinates.

.. note::
   The thesis writes the combination over coordinates as a mean and says the result exceeds 0.5
   exactly when the point is inside the box. For the mean this is false: with a positive box
   size :math:`\sigma(M_c - p_c) + \sigma(p_c - m_c) > 1` in every coordinate, so the mean exceeds
   0.5 everywhere. mOWL uses the minimum over coordinates by default (``aggregate="min"``), which
   has the stated property; ``aggregate="mean"`` is the literal formula.

Losses
------

Every loss is :math:`1 - \mathrm{Vol}(A \cap B)/\mathrm{Vol}(A)`:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Axiom
     - :math:`A`
     - :math:`B`
   * - ``gci0``: :math:`C \sqsubseteq D`
     - :math:`\llbracket C \rrbracket`
     - :math:`\llbracket D \rrbracket`
   * - ``gci1``: :math:`C_1 \sqcap C_2 \sqsubseteq D`
     - :math:`\llbracket C_1 \rrbracket \cap \llbracket C_2 \rrbracket`
     - :math:`\llbracket D \rrbracket`
   * - ``gci2``: :math:`C \sqsubseteq \exists R.D`
     - :math:`\llbracket C \rrbracket`
     - :math:`\pi[\llbracket R \rrbracket \cap (S \times \llbracket D \rrbracket)]`
   * - ``gci3``: :math:`\exists R.C \sqsubseteq D`
     - :math:`\pi[\llbracket R \rrbracket \cap (S \times \llbracket C \rrbracket)]`
     - :math:`\llbracket D \rrbracket`
   * - ``role_inclusion``: :math:`R \sqsubseteq S`
     - :math:`\llbracket R \rrbracket`
     - :math:`\llbracket S \rrbracket`
   * - ``role_chain``: :math:`R_1 \circ R_2 \sqsubseteq S`
     - :math:`\pi_0[\llbracket R_1 \rrbracket] \times \pi_1[\llbracket R_2 \rrbracket]`
     - :math:`\llbracket S \rrbracket`

The role chain loss is the thesis' simplified one and is not fully semantic: the element in the
middle of the chain is not required to be shared. The projection :math:`\pi` is computed exactly
on boxes (a role box contributes its domain box when its codomain box meets the filler) with a
soft overlap test; the thesis does not say how :math:`\pi` is evaluated, so this is our reading.

There are no losses for bottom forms (``gci0_bot``, ``gci1_bot``, ``gci3_bot``), assertions or
negatives, because the thesis defines none. These axioms are skipped with a warning when training.

Evaluation
----------

The distance of Section 7.1.2 is available through ``model.eval_method`` and
``model.evaluation_model``: for ``gci0`` and ``gci1`` the fraction of points sampled from the boxes
of the left concept that lie outside the right concept (the thesis counts the points inside).
Lower is better. For ``gci2`` and ``gci3`` the training loss is used.

Example
-------

The thesis uses :math:`d = 5`, :math:`n = 5`, 20,000 points, 5 epochs and 5 batches (about 8
hours per run on a GPU). The example below uses a tiny budget.

.. testcode::

    from mowl.datasets.builtin import FamilyDataset
    from mowl.models.mcmultibox.model import VolumeMultiBoxEL

    dataset = FamilyDataset()
    model = VolumeMultiBoxEL(dataset, embed_dim=3, num_boxes=2, num_points=256,
                             num_batches=2, learning_rate=0.05, seed=0)
    model.train(epochs=2)
    model.eval_gci_name = "gci0"
    distances = model.eval_method(model.training_datasets["gci0"][:5])
    print(distances.shape)

.. testoutput::
   :options: +ELLIPSIS

   ...torch.Size([5, 1])
