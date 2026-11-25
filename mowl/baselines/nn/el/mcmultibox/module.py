r"""Neural module of the volume-based MultiBoxEL of Aharoni (2024)."""

import functools

import torch as th
import torch.nn as nn
from torch.quasirandom import SobolEngine

import mowl.baselines.nn.el.mcmultibox.volume as V
from mowl.baselines.nn.el.elmodule import ELModule


def _in_row_chunks(space):
    """Compute a per-axiom loss on at most :meth:`VolumeMultiBoxELModule.rows_per_chunk`
    rows at a time. Every loss is computed row by row (one value per axiom), so the result
    is the same; only the size of the largest intermediate tensor changes. ``space`` is
    ``"concept"`` (points in :math:`\\mathbb{R}^d`) or ``"role"`` (:math:`\\mathbb{R}^{2d}`)."""
    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, data, neg=False):
            step = self.rows_per_chunk(space)
            if step is None or len(data) <= step:
                return method(self, data, neg=neg)
            return th.cat([method(self, data[i:i + step], neg=neg).reshape(-1)
                           for i in range(0, len(data), step)])
        return wrapper
    return decorate


class VolumeMultiBoxELModule(ELModule):
    r"""Volume-based MultiBoxEL, after [aharoni2024]_ (Chapter 7).

    Concepts are unions of ``num_boxes`` axis-parallel boxes in :math:`\mathbb{R}^d`, roles are
    unions of ``num_boxes`` boxes in :math:`\mathbb{R}^{2d}` (first ``d`` coordinates: domain,
    last ``d``: codomain) and individuals are single boxes. The weights have the shapes of the
    thesis (Section 7.1): ``individual_embed`` :math:`|O| \times 1 \times 2d`,
    ``class_embed`` :math:`|N_C| \times n \times 2d` and ``rel_embed``
    :math:`|R| \times n \times 4d`, each box being ``[centre, offset]`` (the box is
    ``[centre - |offset|, centre + |offset|]``; the absolute value is ours, so that boxes are
    never empty).

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
        super().__init__()
        if aggregate not in ("mean", "min"):
            raise ValueError("aggregate must be 'mean' or 'min'.")
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.num_boxes = num_boxes
        self.bound = float(bound)
        self.num_points = num_points
        self.temperature = float(temperature)
        self.aggregate = aggregate
        self.seed = seed
        self.point_chunk_size = point_chunk_size
        self.eval_points_per_box = eval_points_per_box
        self.max_chunk_elements = max_chunk_elements

        gen = th.Generator().manual_seed(seed if init_seed is None else init_seed)
        d, n, b = embed_dim, num_boxes, self.bound

        def init(count, boxes, dim):
            centre = (th.rand(count, boxes, dim, generator=gen) * 2 - 1) * b
            offset = (0.4 + 0.5 * th.rand(count, boxes, dim, generator=gen)) * b
            return nn.Parameter(th.cat([centre, offset], dim=-1))

        #: Concepts, shape ``(nb_ont_classes, num_boxes, 2d)``.
        self.class_embed = init(nb_ont_classes, n, d)
        #: Roles, shape ``(nb_rels, num_boxes, 4d)``.
        self.rel_embed = init(nb_rels, n, 2 * d)
        #: Individuals, shape ``(nb_inds, 1, 2d)``; not used by any loss (see class docstring).
        self.ind_embed = init(nb_inds, 1, d) if nb_inds else None

        self.register_buffer("points", V.sobol_points(num_points, d, b, seed), persistent=False)
        self.register_buffer("role_points", V.sobol_points(num_points, 2 * d, b, seed + 1),
                             persistent=False)
        engine = SobolEngine(dimension=d, scramble=True, seed=seed + 2)
        self.register_buffer("eval_unit_points",
                             engine.draw(eval_points_per_box).to(th.float32), persistent=False)

    def rows_per_chunk(self, space="concept"):
        """Axioms per chunk of a loss evaluation (``None``: no limit), from
        :attr:`max_chunk_elements`: the largest tensor is ``rows x points x boxes x k`` with
        ``k = d`` for concept points and ``2d`` for role points."""
        if self.max_chunk_elements is None:
            return None
        k = self.embed_dim * (2 if space == "role" else 1)
        points = min(self.point_chunk_size, self.num_points)
        return max(1, int(self.max_chunk_elements) // (points * self.num_boxes * k))

    # ------------------------------------------------------------------ geometry accessors
    def _boxes(self, table, idx):
        """Centre and (non-negative) offset of the multiboxes ``idx``: two ``(B, n, k)``."""
        w = table[idx]
        k = w.shape[-1] // 2
        return w[..., :k], w[..., k:].abs()

    def concept_boxes(self, idx):
        """Centres and offsets of the concepts ``idx``, each of shape ``(len(idx), n, d)``."""
        return self._boxes(self.class_embed, idx)

    def role_boxes(self, idx):
        """Centres and offsets of the roles ``idx``, each of shape ``(len(idx), n, 2d)``."""
        return self._boxes(self.rel_embed, idx)

    def _chi(self, points, centre, offset):
        return V.soft_membership(points, centre, offset, self.temperature, self.aggregate)

    def _gate(self, role_c, role_o, fill_c, fill_o):
        return V.project_gate(role_c, role_o, fill_c, fill_o, self.bound, self.temperature,
                              self.aggregate)

    def _projection(self, pts, role_c, role_o, gate):
        return V.projection_membership(pts, role_c, role_o, gate, self.temperature,
                                       self.aggregate)

    def _ratio(self, fn, points):
        num, den = V.chunked_means(fn, points, self.point_chunk_size)
        return V.containment_loss(num, den)

    # ---------------------------------------------------------------------------- losses
    @_in_row_chunks("concept")
    def gci0_loss(self, data, neg=False):
        cc, co = self.concept_boxes(data[:, 0])
        dc, do = self.concept_boxes(data[:, 1])

        def fn(p):
            a, b = self._chi(p, cc, co), self._chi(p, dc, do)
            return th.minimum(a, b), a

        return self._ratio(fn, self.points)

    @_in_row_chunks("concept")
    def gci1_loss(self, data, neg=False):
        c1, o1 = self.concept_boxes(data[:, 0])
        c2, o2 = self.concept_boxes(data[:, 1])
        ec, eo = self.concept_boxes(data[:, 2])

        def fn(p):
            a = th.minimum(self._chi(p, c1, o1), self._chi(p, c2, o2))
            return th.minimum(a, self._chi(p, ec, eo)), a

        return self._ratio(fn, self.points)

    @_in_row_chunks("concept")
    def gci2_loss(self, data, neg=False):
        cc, co = self.concept_boxes(data[:, 0])
        rc, ro = self.role_boxes(data[:, 1])
        dc, do = self.concept_boxes(data[:, 2])
        gate = self._gate(rc, ro, dc, do)

        def fn(p):
            a = self._chi(p, cc, co)
            return th.minimum(a, self._projection(p, rc, ro, gate)), a

        return self._ratio(fn, self.points)

    @_in_row_chunks("concept")
    def gci3_loss(self, data, neg=False):
        rc, ro = self.role_boxes(data[:, 0])
        cc, co = self.concept_boxes(data[:, 1])
        dc, do = self.concept_boxes(data[:, 2])
        gate = self._gate(rc, ro, cc, co)

        def fn(p):
            a = self._projection(p, rc, ro, gate)
            return th.minimum(a, self._chi(p, dc, do)), a

        return self._ratio(fn, self.points)

    @_in_row_chunks("role")
    def role_inclusion_loss(self, data, neg=False):
        rc, ro = self.role_boxes(data[:, 0])
        sc, so = self.role_boxes(data[:, 1])

        def fn(p):
            a = self._chi(p, rc, ro)
            return th.minimum(a, self._chi(p, sc, so)), a

        return self._ratio(fn, self.role_points)

    @_in_row_chunks("role")
    def role_chain_loss(self, data, neg=False):
        """Simplified role chain loss of the thesis (not fully semantic, see class docstring)."""
        d = self.embed_dim
        r1c, r1o = self.role_boxes(data[:, 0])
        r2c, r2o = self.role_boxes(data[:, 1])
        sc, so = self.role_boxes(data[:, 2])

        def fn(p):
            x, y = p[:, :d], p[:, d:]
            dom = self._chi(x, r1c[..., :d], r1o[..., :d])
            cod = self._chi(y, r2c[..., d:], r2o[..., d:])
            a = th.minimum(dom, cod)
            return th.minimum(a, self._chi(p, sc, so)), a

        return self._ratio(fn, self.role_points)

    def _unsupported(self, name):
        raise NotImplementedError(
            f"The thesis (Aharoni 2024) defines no loss for '{name}'; "
            f"VolumeMultiBoxELModule does not implement it.")

    def gci0_bot_loss(self, data, neg=False):
        self._unsupported("gci0_bot")

    def gci1_bot_loss(self, data, neg=False):
        self._unsupported("gci1_bot")

    def gci3_bot_loss(self, data, neg=False):
        self._unsupported("gci3_bot")

    def class_assertion_loss(self, data, neg=False):
        self._unsupported("class_assertion")

    def object_property_assertion_loss(self, data, neg=False):
        self._unsupported("object_property_assertion")

    # ------------------------------------------------------------------------ evaluation
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
        if gci_name in ("gci2", "gci3"):
            return self._ranking_ratio(data, gci_name)
        if gci_name not in ("gci0", "gci1"):
            self._unsupported(gci_name)
        outs = []
        for i in range(0, len(data), chunk):
            outs.append(self._sample_distance(data[i:i + chunk], gci_name))
        out = th.cat(outs)
        # one shape for every form: a column, unless ``per_axiom_column`` is off
        return out.unsqueeze(-1) if self.per_axiom_column else out

    def _ranking_ratio(self, data, gci_name):
        """The training loss of ``gci2`` / ``gci3`` rows, computed for ranking.

        The value is the one of :meth:`gci2_loss` / :meth:`gci3_loss` (the volume ratio on the
        fixed point set), but the memberships are computed once per *distinct* operand and
        not once per row: when the candidates of a query differ in one operand only (the
        usual case: one source and many fillers), the membership of the query operand over
        the ``num_points`` points is evaluated once, and so is that of every distinct
        candidate. Points are processed in chunks of :attr:`point_chunk_size`, operands in
        chunks of :meth:`rows_per_chunk`. The cost is that of the distinct operands, not of
        the number of rows times the number of points."""
        if gci_name == "gci2":
            left_keys, right_keys = data[:, :1], data[:, 1:3]  # source C ; (r, D)
        else:
            left_keys, right_keys = data[:, :2], data[:, 2:3]  # (r, C) ; target D
        left_u, left_inv = th.unique(left_keys, dim=0, return_inverse=True)
        right_u, right_inv = th.unique(right_keys, dim=0, return_inverse=True)
        step = self.rows_per_chunk("concept") or max(len(left_u), len(right_u), 1)

        def concept(keys, col):
            return self.concept_boxes(keys[:, col])

        def operand_gci2_left(keys):
            c, o = concept(keys, 0)
            return lambda p: self._chi(p, c, o)

        def projection(keys, role_col, filler_col):
            # the membership in the domain boxes of a role depends on the role only: it is
            # computed once per distinct role; the gate depends on (role, filler) but is small
            roles, inv = th.unique(keys[:, role_col], return_inverse=True)
            rc, ro = self.role_boxes(roles)
            fc, fo = concept(keys, filler_col)
            gate = self._gate(rc[inv], ro[inv], fc, fo)
            d = self.embed_dim

            def fn(p):
                per_box = V.soft_box_membership(p, rc[..., :d], ro[..., :d],
                                                self.temperature, self.aggregate)
                return th.minimum(per_box[inv], gate.unsqueeze(1)).amax(dim=-1)
            return fn

        def operand_gci2_right(keys):
            return projection(keys, 0, 1)

        def operand_gci3_left(keys):
            return projection(keys, 0, 1)

        def operand_gci3_right(keys):
            c, o = concept(keys, 0)
            return lambda p: self._chi(p, c, o)

        left_op, right_op = ((operand_gci2_left, operand_gci2_right) if gci_name == "gci2"
                             else (operand_gci3_left, operand_gci3_right))
        total = self.points.shape[0]
        num = th.zeros(len(data), device=data.device)
        den = th.zeros(len(data), device=data.device)
        for start in range(0, total, self.point_chunk_size):
            p = self.points[start:start + self.point_chunk_size]
            left = th.cat([left_op(left_u[i:i + step])(p) for i in range(0, len(left_u), step)])
            right = th.cat([right_op(right_u[i:i + step])(p)
                            for i in range(0, len(right_u), step)])
            for j in range(0, len(data), step):
                a, b = left[left_inv[j:j + step]], right[right_inv[j:j + step]]
                num[j:j + step] += th.minimum(a, b).sum(dim=-1)
                den[j:j + step] += a.sum(dim=-1)
        out = V.containment_loss(num / total, den / total)
        # the shape the per-axiom loss had (a column unless ``per_axiom_column`` is off)
        return out.unsqueeze(-1) if self.per_axiom_column else out

    def _sample_distance(self, data, gci_name):
        lc, lo = self.concept_boxes(data[:, 0])
        if gci_name == "gci0":
            rc, ro = self.concept_boxes(data[:, 1])
            lo_corner, hi_corner = lc - lo, lc + lo
            valid = th.ones(lc.shape[:2], dtype=th.bool, device=lc.device)
        else:
            c2, o2 = self.concept_boxes(data[:, 1])
            rc, ro = self.concept_boxes(data[:, 2])
            lo_corner = th.maximum((lc - lo).unsqueeze(2), (c2 - o2).unsqueeze(1))
            hi_corner = th.minimum((lc + lo).unsqueeze(2), (c2 + o2).unsqueeze(1))
            valid = (hi_corner >= lo_corner).all(dim=-1).flatten(1)
            k = lc.shape[-1]
            lo_corner, hi_corner = lo_corner.reshape(len(data), -1, k), hi_corner.reshape(
                len(data), -1, k)
        u = self.eval_unit_points.view(1, 1, -1, self.embed_dim)
        pts = lo_corner.unsqueeze(2) + (hi_corner - lo_corner).unsqueeze(2) * u  # (B, nb, k, d)
        pts = pts.flatten(1, 2)  # (B, nb*k, d)
        w = valid.unsqueeze(-1).expand(-1, -1, u.shape[2]).flatten(1)
        rlo, rhi = (rc - ro).unsqueeze(1), (rc + ro).unsqueeze(1)
        inside = (pts.unsqueeze(2) >= rlo) & (pts.unsqueeze(2) <= rhi)
        inside = inside.all(dim=-1).any(dim=-1)  # (B, nb*k)
        hits = (inside & w).sum(dim=-1).to(th.float32)
        total = w.sum(dim=-1).to(th.float32)
        return th.where(total > 0, 1.0 - hits / total.clamp(min=1.0), th.ones_like(hits))
