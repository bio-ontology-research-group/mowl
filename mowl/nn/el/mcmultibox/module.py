r"""Neural module of the volume-based MultiBoxEL of Aharoni (2024)."""

import torch as th
from torch.quasirandom import SobolEngine

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.multibox_distance import sampled_distance
from mowl.representation.library.multibox_methods import mcmultibox
from mowl.representation.library.multibox_modules import ReleasedTables, released_names

#: The forms the thesis defines a loss for.
_FORMS = ("gci0", "gci1", "gci2", "gci3", "role_inclusion", "role_chain")


@released_names(class_embed="class_model.centres", rel_embed="relation_model.vectors",
                ind_embed="individual_model.centres")
class VolumeMultiBoxELModule(ReleasedTables, ComposedGeometry):
    r"""Volume-based MultiBoxEL, after [aharoni2024]_ (Chapter 7).

    Concepts are unions of ``num_boxes`` axis-parallel boxes in :math:`\mathbb{R}^d`, roles are
    unions of ``num_boxes`` boxes in :math:`\mathbb{R}^{2d}` (first ``d`` coordinates: domain,
    last ``d``: codomain) and individuals are single boxes. The weights have the shapes of the
    thesis (Section 7.1): ``individual_embed`` :math:`|O| \times 1 \times 2d`,
    ``class_embed`` :math:`|N_C| \times n \times 2d` and ``rel_embed``
    :math:`|R| \times n \times 4d`, each box being ``[centre, offset]`` (the box is
    ``[centre - |offset|, centre + |offset|]``; the absolute value is ours, so that boxes are
    never empty).

    The module is the combination of parts that the volume-based MultiBoxEL is
    (:func:`~mowl.representation.library.multibox_methods.mcmultibox`): the class model
    :class:`~mowl.representation.library.multibox_parts.PackedUnionClasses` (``class_embed``),
    the relation model :class:`~mowl.representation.library.multibox_parts.PairUnions`
    (``rel_embed``), the measure
    :class:`~mowl.representation.library.multibox_parts.SampledVolume` (the point sets,
    ``temperature``, ``aggregate``) and the semantic objective. It is a
    :class:`~mowl.representation.ComposedGeometry`; the evaluation distance of Section 7.1.2
    (:meth:`distance`) is
    :func:`~mowl.representation.library.multibox_distance.sampled_distance` on its parts.

    Volumes are estimated on a fixed set of ``num_points`` quasi-random points of
    :math:`S = [-b_o, b_o]^d` (a scrambled Sobol sequence, deterministic given ``seed``; a second
    set in :math:`[-b_o, b_o]^{2d}` is used for the role losses) with the soft membership
    function of :mod:`mowl.nn.el.mcmultibox.volume`. Every loss is
    :math:`1 - \mathrm{Vol}(A \cap B)/\mathrm{Vol}(A)`:

    * ``gci0`` :math:`C \sqsubseteq D`: :math:`A = \llbracket C\rrbracket`,
      :math:`B = \llbracket D\rrbracket`.
    * ``gci1`` :math:`C_1 \sqcap C_2 \sqsubseteq D`: :math:`A = \llbracket C_1\rrbracket \cap
      \llbracket C_2\rrbracket` (union of pairwise box intersections).
    * ``gci2`` :math:`C \sqsubseteq \exists R.D`: :math:`B = \pi[\llbracket R\rrbracket \cap
      (S \times \llbracket D\rrbracket)]`.
    * ``gci3`` :math:`\exists R.C \sqsubseteq D`: :math:`A = \pi[\llbracket R\rrbracket \cap
      (S \times \llbracket C\rrbracket)]`.
    * ``role_inclusion`` :math:`R \sqsubseteq S`: :math:`A = \llbracket R\rrbracket`,
      :math:`B = \llbracket S\rrbracket` in :math:`\mathbb{R}^{2d}`.
    * ``role_chain`` :math:`R_1 \circ R_2 \sqsubseteq S`: **the thesis' simplified loss**, not a
      semantically complete one: :math:`A = \pi_0[\llbracket R_1\rrbracket] \times
      \pi_1[\llbracket R_2\rrbracket]` (domain of :math:`R_1` times codomain of :math:`R_2`),
      :math:`B = \llbracket S\rrbracket`. It ignores that the middle element of the chain must be
      shared, so it can be zero for a configuration that does not satisfy the chain axiom.

    The projection :math:`\pi` (onto the domain) is computed exactly on boxes, not by sampling:
    :math:`\pi[R_j \cap (S \times D)]` is the domain box of :math:`R_j` if its codomain box meets
    the filler multibox (inside :math:`S`) and empty otherwise. The "meets" test is made soft with
    the same sigmoid construction as the membership function (see
    :func:`~mowl.nn.el.mcmultibox.volume.soft_overlap_gate`). This is our reading of the
    thesis' :math:`\pi`; the thesis does not describe how it is evaluated.

    **Not covered.** The thesis defines no loss for the bottom forms (``gci0_bot``, ``gci1_bot``,
    ``gci3_bot``), for class or role assertions, or for negative sampling, so none is
    implemented: :attr:`neg_capable_gcis` is empty and those losses raise
    :class:`NotImplementedError`. The individual weights :math:`A` exist (shape as in the
    thesis) but no loss uses them. Concepts that the ontology makes unsatisfiable are therefore
    not driven to the empty set.

    **Deviations / unspecified in the thesis** (ours): the bound :math:`b_o` (default 1), the
    aggregate over coordinates (``"min"`` by default, because the thesis' literal ``mean`` is
    positive everywhere, see :mod:`mowl.nn.el.mcmultibox.volume`), the
    initialisation (centres uniform in :math:`[-b_o, b_o]`, offsets uniform in
    :math:`[0.4, 0.9] b_o`, large enough that random multiboxes overlap in
    :math:`d = 5`), the sigmoid temperature (1, as the thesis), set intersection as
    pointwise minimum of memberships, the :math:`\varepsilon` in the denominator, and the use of
    a scrambled Sobol sequence. Because the volume estimate uses the soft membership, which is
    below 0.5, all volumes are scaled by the same factor; ratios and losses are unaffected.

    :param nb_ont_classes: Number of concepts (named and auxiliary).
    :param nb_rels: Number of roles (named and auxiliary).
    :param nb_inds: Number of individuals, or ``None``.
    :param embed_dim: Dimension :math:`d` of the concept space (thesis: 5).
    :param num_boxes: Boxes per multibox :math:`n` (thesis: 5).
    :param bound: Half-width :math:`b_o` of the space :math:`S`.
    :param num_points: Number of quasi-Monte-Carlo points ``#QMC`` (thesis: 20,000; the thesis
        reports that fewer than 10,000 gave zero evaluation scores).
    :param temperature: Sigmoid temperature (thesis: none, i.e. 1).
    :param aggregate: ``"min"`` (default; membership positive iff inside) or ``"mean"`` (the
        literal thesis formula, positive everywhere; see :mod:`mowl.nn.el.mcmultibox.volume`).
    :param seed: Seed of the point set (and of the weight initialisation if ``init_seed`` is
        ``None``).
    :param point_chunk_size: Number of points processed at once (memory/speed trade-off only).
    :param eval_points_per_box: Points sampled per box of the left-hand concept by the
        evaluation distance of Section 7.1.2.
    :param max_chunk_elements: Largest number of elements of one intermediate tensor
        (axioms x points x boxes x coordinates) of a loss: the axioms of a batch are
        processed :meth:`rows_per_chunk` at a time. The numbers do not change; the memory
        does. With the default (``2**25``, 128 MiB per tensor in float32) one loss evaluation
        holds about 16 such tensors at its peak under gradients (about 2 GiB), whatever the
        batch size. ``None``: the whole batch at once (memory grows with the batch: about
        200 KiB per axiom and tensor with the thesis settings, so a batch of 24,000 axioms
        needs tens of GiB).
    """

    #: The thesis defines no negative losses.
    neg_capable_gcis = frozenset()
    role_axiom_capable = True

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=5, num_boxes=5,
                 bound=1.0, num_points=20000, temperature=1.0, aggregate="min", seed=0,
                 point_chunk_size=2048, eval_points_per_box=32, init_seed=None,
                 max_chunk_elements=2 ** 25):
        if aggregate not in ("mean", "min"):
            raise ValueError("aggregate must be 'mean' or 'min'.")
        mcmultibox(nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim,
                   num_boxes=num_boxes, bound=bound, num_points=num_points,
                   temperature=temperature, aggregate=aggregate, seed=seed,
                   point_chunk_size=point_chunk_size, init_seed=init_seed, into=self)
        self.supported_forms = frozenset(_FORMS)
        self.strict_negatives = True
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.num_boxes = num_boxes
        self.num_points = num_points
        self.seed = seed
        self.eval_points_per_box = eval_points_per_box
        self.max_chunk_elements = max_chunk_elements
        engine = SobolEngine(dimension=embed_dim, scramble=True, seed=seed + 2)
        self.register_buffer("eval_unit_points",
                             engine.draw(eval_points_per_box).to(th.float32), persistent=False)

    # -- the numbers of the method, which are those of the parts -----------------------
    @property
    def bound(self):
        """Half-width :math:`b_o` of the space."""
        return self.measure.bound

    @bound.setter
    def bound(self, value):
        self.measure.bound = float(value)
        self.relation_model.bound = float(value)

    @property
    def temperature(self):
        """Temperature of the sigmoid of the membership."""
        return self.measure.temperature

    @temperature.setter
    def temperature(self, value):
        self.measure.temperature = float(value)

    @property
    def aggregate(self):
        """``"min"`` or ``"mean"`` over the coordinates of the membership."""
        return self.measure.aggregate

    @aggregate.setter
    def aggregate(self, value):
        if value not in ("mean", "min"):
            raise ValueError("aggregate must be 'mean' or 'min'.")
        self.measure.aggregate = value

    @property
    def point_chunk_size(self):
        """Points processed at a time."""
        return self.measure.chunk_size

    @point_chunk_size.setter
    def point_chunk_size(self, value):
        self.measure.chunk_size = value

    @property
    def points(self):
        """The point set of the concept space ``(num_points, d)``."""
        return self.measure.points(self.embed_dim)

    @property
    def role_points(self):
        """The point set of the space of pairs ``(num_points, 2d)``."""
        return self.measure.points(2 * self.embed_dim)

    def rows_per_chunk(self, space="concept"):
        """Axioms per chunk of a loss evaluation (``None``: no limit), from
        :attr:`max_chunk_elements`: the largest tensor is ``rows x points x boxes x k`` with
        ``k = d`` for concept points and ``2d`` for role points."""
        if self.max_chunk_elements is None:
            return None
        k = self.embed_dim * (2 if space == "role" else 1)
        points = min(self.point_chunk_size, self.num_points)
        return max(1, int(self.max_chunk_elements) // (points * self.num_boxes * k))

    # -- geometry accessors ------------------------------------------------------------
    def concept_boxes(self, idx):
        """Centres and offsets of the concepts ``idx``, each of shape ``(len(idx), n, d)``."""
        region = self.classes(idx)
        return region.centre, region.offset

    def role_boxes(self, idx):
        """Centres and offsets of the roles ``idx``, each of shape ``(len(idx), n, 2d)``."""
        return self.relation_model._boxes(idx)

    # -- losses ----------------------------------------------------------------------------
    def cost(self, form, rows, neg=False):
        if form not in _FORMS:
            raise NotImplementedError(
                f"The thesis (Aharoni 2024) defines no loss for '{form.replace('_loss', '')}'"
                f"; VolumeMultiBoxELModule does not implement it.")
        step = self.rows_per_chunk("role" if form.startswith("role") else "concept")
        if step is None or len(rows) <= step:
            return super().cost(form, rows, neg)
        return th.cat([super(VolumeMultiBoxELModule, self).cost(form, rows[i:i + step], neg)
                       for i in range(0, len(rows), step)])

    def _unsupported(self, name):
        raise NotImplementedError(
            f"The thesis (Aharoni 2024) defines no loss for '{name}'; "
            f"VolumeMultiBoxELModule does not implement it.")

    # -- evaluation ------------------------------------------------------------------------
    @th.no_grad()
    def distance(self, data, gci_name, chunk=1024):
        r"""Evaluation distance of Section 7.1.2 (lower is better).

        For ``gci0`` (:math:`C \sqsubseteq D`) it is :math:`1 - |\{p \in P_C : p \in \llbracket
        D\rrbracket\}|/|P_C|`, where :math:`P_C` holds ``eval_points_per_box`` points drawn
        (quasi-uniformly, deterministically) from each box of :math:`C`, and membership in
        :math:`D` is exact. The thesis writes the count of contained points; we divide by the
        number of points and negate so that, as for every score in mOWL, smaller means better.
        For ``gci1`` the points are drawn from the non-empty pairwise intersections of the boxes
        of the two left concepts. For ``gci2`` and ``gci3`` the thesis does not give a distance;
        we use the training loss (volume ratio on the fixed point set). Many pairs have
        distance exactly 1, which is how the thesis obtains Hits@10 close to 0.

        The result has shape ``(N, 1)`` for every form (``(N,)`` with ``per_axiom_column =
        False``).

        :param data: ``(N, 2)`` for ``gci0``, ``(N, 3)`` otherwise.
        :param gci_name: ``"gci0"``, ``"gci1"``, ``"gci2"`` or ``"gci3"``.
        """
        if gci_name not in ("gci0", "gci1", "gci2", "gci3"):
            self._unsupported(gci_name)
        out = sampled_distance(self, data, gci_name, self.eval_unit_points, chunk=chunk,
                               step=self.rows_per_chunk("concept"))
        return out.unsqueeze(-1) if self.per_axiom_column else out
