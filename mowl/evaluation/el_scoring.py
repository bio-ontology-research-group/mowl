"""Scoring functions for candidate axioms of :math:`\\mathcal{EL}^{++}` embedding models.

An evaluation regime ranks candidate axioms by a score. Two conventions are used in the
literature:

* **loss scores**: the model's own loss for the normal form is the score (lower is
  "more true"). This is what DELE does ("we use the corresponding loss expressions to
  score axioms", Mashkova et al., Section 6.2) and what mOWL's evaluators do.
* **centre-distance scores** of Box\\ :sup:`2`\\ EL (Jackermeier et al., WWW 2024,
  Appendix F): the distance between the centres of the regions on both sides of the
  subsumption, with model-specific formulas for :math:`C \\sqsubseteq \\exists r.D` and
  :math:`\\exists r.C \\sqsubseteq D` (bump vectors for Box\\ :sup:`2`\\ EL, translations
  for the translational baselines ELEm, EmEL++ and ELBE; TransBox's own formulas, which
  mask the dimensions of an empty intersection; for BoxEL one minus an overlap volume,
  as in the Box\\ :sup:`2`\\ EL code). Published Box\\ :sup:`2`\\ EL-protocol numbers
  were obtained with these scores.
* **centre distance, zero on empty intersection**: the variant of the TransBox paper
  (Yang et al., WWW 2025, Appendix E) for :math:`C \\sqcap D \\sqsubseteq E`; see
  :func:`centre_distance_zero_empty_scores`.

All scorers take rows in mOWL's column layout (``gci0``: ``[C, D]``; ``gci1``:
``[C, D, E]``; ``gci2``: ``[C, r, D]``; ``gci3``: ``[r, C, D]``; ``class_assertion``:
``[a, C]``) and return one score per row, lower is better. Centre-distance scorers are
registered per module class in :data:`CENTRE_DISTANCE_SCORERS`; to add one for a new
module, register a function ``f(module, gci_name, rows) -> tensor`` under the module's
class name.
"""

import torch as th


def loss_scores(model, gci_name, rows):
    """Score rows with the model's loss for ``gci_name`` (``model.module(rows, gci_name)``).

    :param model: A trained :class:`~mowl.base_models.EmbeddingELModel` (anything with a
        ``module`` callable as ``module(rows, gci_name)``).
    """
    out = model.module(rows, gci_name)
    return out.reshape(len(rows), -1).mean(dim=1)


def margin_free_loss_scores(model, gci_name, rows):
    r"""Score rows with the model's *margin-free* loss (``module.zero_margin_loss``).

    This is the test-time score of ELK-Em (Singh et al. 2026, Eq. 4: the score of an
    axiom is :math:`s(\alpha) = e^{-\ell^0(\alpha)}` with :math:`\ell^0` the per-axiom
    loss with margin zero; the exponential is monotone, so ranking by :math:`\ell^0`
    ranks by :math:`s`). The reference evaluation code scores positives and corruptions
    with the margin set to zero while training uses :math:`\gamma = 0.1`. Modules
    without ``zero_margin_loss`` fall back to their ordinary loss.
    """
    module = model.module
    if hasattr(module, "zero_margin_loss"):
        out = module.zero_margin_loss(rows, gci_name)
    else:
        out = module(rows, gci_name)
    return out.reshape(len(rows), -1).mean(dim=1)


def _box_intersection_centres(c1, o1, c2, o2):
    lower = th.maximum(c1 - o1, c2 - o2)
    upper = th.minimum(c1 + o1, c2 + o2)
    return (lower + upper) / 2


def _box2el_scores(module, gci_name, rows):
    centre = module.class_center.weight
    offset = module.class_offset.weight
    bump = module.bump_classes.weight
    head = module.head_center.weight
    tail = module.tail_center.weight
    if gci_name == "gci0":
        return th.linalg.norm(centre[rows[:, 0]] - centre[rows[:, 1]], dim=1)
    if gci_name == "gci1":
        inter = _box_intersection_centres(centre[rows[:, 0]], th.abs(offset[rows[:, 0]]),
                                          centre[rows[:, 1]], th.abs(offset[rows[:, 1]]))
        return th.linalg.norm(inter - centre[rows[:, 2]], dim=1)
    if gci_name == "gci2":
        c, r, d = rows[:, 0], rows[:, 1], rows[:, 2]
        to_head = th.linalg.norm(centre[c] + bump[d] - head[r], dim=1)
        to_tail = th.linalg.norm(centre[d] + bump[c] - tail[r], dim=1)
        return to_head + to_tail
    if gci_name == "gci3":
        r, c, d = rows[:, 0], rows[:, 1], rows[:, 2]
        return th.linalg.norm(head[r] - bump[c] - centre[d], dim=1)
    raise NotImplementedError(gci_name)


def _translational_scores(centre, half_width, rel, gci_name, rows):
    if gci_name == "gci0":
        return th.linalg.norm(centre[rows[:, 0]] - centre[rows[:, 1]], dim=1)
    if gci_name == "gci1":
        inter = _box_intersection_centres(centre[rows[:, 0]], half_width(rows[:, 0]),
                                          centre[rows[:, 1]], half_width(rows[:, 1]))
        return th.linalg.norm(inter - centre[rows[:, 2]], dim=1)
    if gci_name == "gci2":
        c, r, d = rows[:, 0], rows[:, 1], rows[:, 2]
        return th.linalg.norm(centre[c] + rel[r] - centre[d], dim=1)
    if gci_name == "gci3":
        r, c, d = rows[:, 0], rows[:, 1], rows[:, 2]
        return th.linalg.norm(centre[c] - rel[r] - centre[d], dim=1)
    raise NotImplementedError(gci_name)


def _elbe_scores(module, gci_name, rows):
    centre = module.class_embed.weight
    offset = module.class_offset.weight
    return _translational_scores(centre, lambda i: th.abs(offset[i]),
                                 module.rel_embed.weight, gci_name, rows)


def _elem_scores(module, gci_name, rows):
    centre = module.class_embed.weight
    radius = module.class_rad.weight
    # Balls are treated as boxes with half-width |radius| in every coordinate for the
    # intersection in C ⊓ D ⊑ E, as in the Box2EL baselines (see Yang et al., WWW 2025,
    # Appendix E).
    return _translational_scores(centre,
                                 lambda i: th.abs(radius[i]).expand(-1, centre.shape[1]),
                                 module.rel_embed.weight, gci_name, rows)


def _transbox_scores(module, gci_name, rows):
    """Centre distances of the TransBox reference code (``evaluate.py``,
    ``compute_nf*_ranks_transbox``): :math:`\\exists r.D` has centre
    :math:`c(r) + c(D)`, and for :math:`C \\sqcap D \\sqsubseteq E` only the dimensions
    in which the boxes of :math:`C` and :math:`D` intersect are compared."""
    centre = module.class_center_embedding.weight
    offset = module.class_offset_embedding.weight.abs()
    rel = module.relation_center_embedding.weight
    if gci_name == "gci0":
        return th.linalg.norm(centre[rows[:, 0]] - centre[rows[:, 1]], dim=1)
    if gci_name == "gci1":
        c, d = rows[:, 0], rows[:, 1]
        lower = th.maximum(centre[c] - offset[c], centre[d] - offset[d])
        upper = th.minimum(centre[c] + offset[c], centre[d] + offset[d])
        mask = (lower <= upper).to(centre.dtype)
        return th.linalg.norm(((lower + upper) / 2 - centre[rows[:, 2]]) * mask, dim=1)
    if gci_name == "gci2":
        c, r, d = rows[:, 0], rows[:, 1], rows[:, 2]
        return th.linalg.norm(rel[r] + centre[d] - centre[c], dim=1)
    if gci_name == "gci3":
        r, c, d = rows[:, 0], rows[:, 1], rows[:, 2]
        return th.linalg.norm(rel[r] + centre[c] - centre[d], dim=1)
    raise NotImplementedError(gci_name)


def boxel_overlap_scores(module, gci_name, rows):
    """BoxEL in the Box2EL code (``evaluate.py``, ``compute_nf*_ranks_boxel``): one minus
    the soft volume of the intersection of the two sides (divided by the volume of the
    transformed left-hand side for :math:`C \\sqsubseteq \\exists r.D`). Not a centre
    distance, but the score the published BoxEL baseline numbers were obtained with.

    The reference function for :math:`\\exists r.C \\sqsubseteq D` ranks the class whose
    index equals the *role* index (``batch_data[:, 0]``); here the score is the same but
    the protocol ranks :math:`D`, so that normal form is not comparable with the
    published BoxEL numbers."""
    import torch.nn.functional as F
    lo_all = module.min_embedding.weight
    hi_all = lo_all + th.exp(module.delta_embedding.weight)
    rel = module.relation_embedding.weight
    scale = module.scaling_embedding.weight

    def volume(lo, hi):
        return th.clamp(F.softplus(hi - lo).prod(1), 1e-10, 1e4)

    def inter(lo1, hi1, lo2, hi2):
        return th.maximum(lo1, lo2), th.minimum(hi1, hi2)

    if gci_name == "gci0":
        c, d = rows[:, 0], rows[:, 1]
        return 1 - volume(*inter(lo_all[c], hi_all[c], lo_all[d], hi_all[d]))
    if gci_name == "gci1":
        c, d, e = rows[:, 0], rows[:, 1], rows[:, 2]
        lo, hi = inter(lo_all[c], hi_all[c], lo_all[d], hi_all[d])
        return 1 - volume(*inter(lo, hi, lo_all[e], hi_all[e]))
    if gci_name == "gci2":
        c, r, d = rows[:, 0], rows[:, 1], rows[:, 2]
        t_lo = lo_all[c] * (scale[r] + 1e-8) + rel[r]
        t_hi = hi_all[c] * (scale[r] + 1e-8) + rel[r]
        return 1 - volume(*inter(lo_all[d], hi_all[d], t_lo, t_hi)) / volume(t_lo, t_hi)
    if gci_name == "gci3":
        r, c, d = rows[:, 0], rows[:, 1], rows[:, 2]
        t_lo = (lo_all[c] - rel[r]) / (scale[r] + 1e-8)
        t_hi = (hi_all[c] - rel[r]) / (scale[r] + 1e-8)
        return 1 - volume(*inter(t_lo, t_hi, lo_all[d], hi_all[d]))
    raise NotImplementedError(gci_name)


#: Centre-distance scorers of Jackermeier et al. (WWW 2024, Appendix F), by module class
#: name. Box2EL uses its bump-vector formulas, ELBE and ELEm the translational ones.
# Reference formulas by definition: each entry is the published score of one module class.
CENTRE_DISTANCE_SCORERS = {
    "BoxSquaredELModule": _box2el_scores,
    "ELBEModule": _elbe_scores,
    "RegDELBEModule": _elbe_scores,   # (a combination of parts, not a subclass of ELBEModule)
    "ELEmModule": _elem_scores,
    "TransBoxModule": _transbox_scores,
    "BoxELModule": boxel_overlap_scores,
}


#: Scorers whose score already treats empty intersections in the way of their own paper.
_OWN_EMPTY_INTERSECTION_RULE = (_transbox_scores, boxel_overlap_scores)


def _scorer_of(model):
    """The scorer registered for the module's class or the nearest base class (EmEL++
    is scored like ELEm, as in the Box2EL code)."""
    for cls in type(getattr(model, "module", None)).__mro__:
        if cls.__name__ in CENTRE_DISTANCE_SCORERS:
            return CENTRE_DISTANCE_SCORERS[cls.__name__]
    return None


def has_centre_distance_scorer(model):
    """Whether :data:`CENTRE_DISTANCE_SCORERS` has an entry for the model's module."""
    return _scorer_of(model) is not None


def centre_distance_scores(model, gci_name, rows):
    """Centre-distance score of Box\\ :sup:`2`\\ EL Appendix F for the model's module.

    :raises NotImplementedError: If no scorer is registered for the module.
    """
    name = type(model.module).__name__
    scorer = _scorer_of(model)
    if scorer is None:
        raise NotImplementedError(
            f"No centre-distance scorer is registered for {name}. Register one in "
            f"mowl.evaluation.el_scoring.CENTRE_DISTANCE_SCORERS or use scoring='loss'.")
    return scorer(model.module, gci_name, rows)


def empty_intersection(model, rows):
    """For rows ``[C, D, ...]``: whether the boxes of ``C`` and ``D`` are disjoint (in at
    least one dimension), or ``None`` if the classes of the model are not boxes or balls
    or it has no geometry yet. The regions are those of the geometry of the model
    (:func:`mowl.representation.geometry_of`); balls count as boxes with the radius as
    half-width in every dimension, as in the Box2EL code."""
    from mowl.evaluation.region_access import all_class_regions, centre_and_size
    try:
        region, _ = all_class_regions(model, "The empty-intersection rule")
        centre, half, is_ball = centre_and_size(region, "The empty-intersection rule")
    except (NotImplementedError, TypeError):
        return None
    if is_ball:
        half = half[:, None].expand(-1, centre.shape[1])
    c, d = rows[:, 0], rows[:, 1]
    lower = th.maximum(centre[c] - half[c], centre[d] - half[d])
    upper = th.minimum(centre[c] + half[c], centre[d] + half[d])
    return (lower > upper).any(dim=1)


def centre_distance_zero_empty_scores(model, gci_name, rows):
    """:func:`centre_distance_scores` with the rule of the TransBox evaluation (Yang et
    al., WWW 2025, Appendix E; ``evaluate.py``, ``compute_nf2_ranks``): for
    :math:`C \\sqcap D \\sqsubseteq E`, every candidate gets score 0 when the boxes of
    :math:`C` and :math:`D` do not intersect. All candidates then tie, and the rank of
    the true answer is its position in the candidate list. Models with a scorer that
    handles empty intersections itself (TransBox masks the empty dimensions; BoxEL's is a
    volume) and models whose classes are not boxes or balls are scored as by
    :func:`centre_distance_scores`."""
    scores = centre_distance_scores(model, gci_name, rows)
    if gci_name != "gci1" or _scorer_of(model) in _OWN_EMPTY_INTERSECTION_RULE:
        return scores
    empty = empty_intersection(model, rows)
    if empty is None:
        return scores
    return th.where(empty, th.zeros_like(scores), scores)


def resolve_scorer(model, scoring):
    """Return ``(scorer, name)`` for a scoring choice.

    :param scoring: ``"loss"``, ``"centre_distance"``, ``"centre_distance_zero_empty"``
        (see :func:`centre_distance_zero_empty_scores`), ``"auto"`` (centre distance if a
        scorer is registered for the module; otherwise what the model declares with its
        ``axiom_scoring`` attribute, see :mod:`mowl.evaluation.model_scoring`: the loss of
        its module, its own axiom costs, or a classifier head on its vectors),
        ``"model"`` (what the model declares, never the centre distance),
        ``"pair_classifier"`` (a :class:`~mowl.evaluation.model_scoring.PairClassifierHead`
        on the class vectors, for any model with ``class_embeddings``) or a callable
        ``f(model, gci_name, rows)``.
    :return: The function and the name of the scoring actually used (recorded in results).
    """
    if callable(scoring):
        return scoring, getattr(scoring, "__name__", "custom")
    if scoring == "loss":
        return loss_scores, "loss"
    if scoring == "margin_free":
        return margin_free_loss_scores, "margin_free"
    if scoring == "centre_distance":
        return centre_distance_scores, "centre_distance"
    if scoring == "centre_distance_zero_empty":
        return centre_distance_zero_empty_scores, "centre_distance_zero_empty"
    if scoring == "auto":
        if has_centre_distance_scorer(model):
            return centre_distance_scores, "centre_distance"
        if getattr(model, "axiom_scoring", None) in (None, "module") \
                and getattr(model, "module", None) is not None:
            return loss_scores, "loss"
        from mowl.evaluation.model_scoring import resolve_model_scorer
        return resolve_model_scorer(model)
    if scoring == "model":
        from mowl.evaluation.model_scoring import resolve_model_scorer
        return resolve_model_scorer(model)
    if scoring == "pair_classifier":
        from mowl.evaluation.model_scoring import pair_classifier_scores
        return pair_classifier_scores, "pair_classifier"
    raise ValueError(f"Unknown scoring {scoring!r}.")
