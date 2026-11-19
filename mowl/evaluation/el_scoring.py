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
  for the translational baselines ELEm and ELBE). Published Box\\ :sup:`2`\\ EL-protocol
  numbers were obtained with these scores.

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


#: Centre-distance scorers of Jackermeier et al. (WWW 2024, Appendix F), by module class
#: name. Box2EL uses its bump-vector formulas, ELBE and ELEm the translational ones.
CENTRE_DISTANCE_SCORERS = {
    "BoxSquaredELModule": _box2el_scores,
    "ELBEModule": _elbe_scores,
    "ELEmModule": _elem_scores,
}


def has_centre_distance_scorer(model):
    """Whether :data:`CENTRE_DISTANCE_SCORERS` has an entry for the model's module."""
    return type(getattr(model, "module", None)).__name__ in CENTRE_DISTANCE_SCORERS


def centre_distance_scores(model, gci_name, rows):
    """Centre-distance score of Box\\ :sup:`2`\\ EL Appendix F for the model's module.

    :raises NotImplementedError: If no scorer is registered for the module.
    """
    name = type(model.module).__name__
    if name not in CENTRE_DISTANCE_SCORERS:
        raise NotImplementedError(
            f"No centre-distance scorer is registered for {name}. Register one in "
            f"mowl.evaluation.el_scoring.CENTRE_DISTANCE_SCORERS or use scoring='loss'.")
    return CENTRE_DISTANCE_SCORERS[name](model.module, gci_name, rows)


def resolve_scorer(model, scoring):
    """Return ``(scorer, name)`` for a scoring choice.

    :param scoring: ``"loss"``, ``"centre_distance"``, ``"auto"`` (centre distance if a
        scorer is registered for the module, otherwise loss) or a callable
        ``f(model, gci_name, rows)``.
    :return: The function and the name of the scoring actually used (recorded in results).
    """
    if callable(scoring):
        return scoring, getattr(scoring, "__name__", "custom")
    if scoring == "loss":
        return loss_scores, "loss"
    if scoring == "centre_distance":
        return centre_distance_scores, "centre_distance"
    if scoring == "auto":
        if has_centre_distance_scorer(model):
            return centre_distance_scores, "centre_distance"
        return loss_scores, "loss"
    raise ValueError(f"Unknown scoring {scoring!r}.")
