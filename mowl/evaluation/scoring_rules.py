"""A third scoring rule for ranking candidate axioms: the margin-free loss.

:mod:`mowl.evaluation.el_scoring` provides the two scoring rules of the literature that
mOWL had so far: the model's loss (DELE; mOWL's evaluators) and the centre distance of
Box\\ :sup:`2`\\ EL (Appendix F; TransBox, Appendix F; Aharoni 2024, Section 7.1.2; and the
"distance from :math:`D`" of EmEL++, Section 4.4). ELK-Em (Singh et al. 2026, Section 4.2,
Eq. 4) ranks by :math:`s(\\alpha) = e^{-\\ell_0(\\alpha)}`, where :math:`\\ell_0` is the
per-axiom loss with the margin set to 0. Because :math:`e^{-x}` is monotone, this ranks
exactly like the loss *with margin 0*, which differs from the loss with the trained
margin whenever the margin is non-zero (with :math:`\\gamma > 0` the hinge
:math:`\\max(0, d - \\gamma)` assigns the same score 0 to every candidate within the
margin, creating ties).

:func:`margin_free_loss_scores` is a scorer for
:func:`mowl.evaluation.el_scoring.resolve_scorer` (pass it as ``scoring=``). It sets the
module attributes ``margin`` and ``gamma`` (the names used by mOWL's modules) to 0 while
scoring and restores them afterwards.
"""

import torch as th

from mowl.evaluation.el_scoring import loss_scores

MARGIN_ATTRIBUTES = ("margin", "gamma")


def margin_free_loss_scores(model, gci_name, rows):
    """Loss scores with the module's margin temporarily set to 0 (lower is better).

    :raises ValueError: If the module has neither a ``margin`` nor a ``gamma`` attribute.
    """
    module = model.module
    saved = {a: getattr(module, a) for a in MARGIN_ATTRIBUTES if hasattr(module, a)}
    if not saved:
        raise ValueError(f"{type(module).__name__} has no 'margin' or 'gamma' attribute.")
    try:
        for a in saved:
            setattr(module, a, 0.0)
        return loss_scores(model, gci_name, rows)
    finally:
        for a, v in saved.items():
            setattr(module, a, v)


def exp_negative_score(loss):
    """Score :math:`e^{-\\ell} \\in (0, 1]` of a loss :math:`\\ell \\ge 0`: higher is better,
    1 for an axiom that holds. A monotone map, so it ranks like the loss."""
    return th.exp(-loss)
