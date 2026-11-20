"""The meaning of the margin of an inclusion loss, as a training regime.

Region-based :math:`\\mathcal{EL}^{++}` losses contain a margin :math:`\\gamma`, and
implementations disagree on its sign. For the inclusion of a box with centre
:math:`c_1` and half-width :math:`o_1` in a box :math:`(c_2, o_2)`:

* **strict**: :math:`\\lVert\\max(0, |c_1 - c_2| + o_1 - o_2 + \\gamma)\\rVert`. The loss
  is zero only if the first box lies inside the second with a gap of at least
  :math:`\\gamma` on every side. Zero loss implies the axiom holds. This is the loss of
  the ELBE paper and code (Peng et al. 2022) and of the TransBox paper (Yang et al.
  2025, Section 4.3).
* **slack**: :math:`\\lVert\\max(0, |c_1 - c_2| + o_1 - o_2 - \\gamma)\\rVert`. The loss is
  already zero when the first box sticks out of the second by up to :math:`\\gamma` in
  every dimension. Zero loss does *not* imply the axiom holds, so a trained embedding is
  in general not a model of the ontology. This is the loss of ELEmbeddings, of
  Box\\ :sup:`2`\\ EL (Jackermeier et al. 2024, Section 3.2), of the ELBE baseline in the
  Box\\ :sup:`2`\\ EL code, and of the TransBox code.

The slack form suits evaluations that rank by the distance between centres
(:func:`mowl.evaluation.el_scoring.centre_distance_scores`), which ignore the sizes of
the regions: it lets training pull centres together without having to nest the boxes.

Every mOWL module declares which form its ``margin`` parameter has
(:attr:`mowl.nn.ELModule.margin_semantics`). The regimes below set the margin by
meaning, independently of the module's sign convention.
"""

from mowl.training.regime import TrainingRegime


def margin_semantics(model):
    """``"strict"`` or ``"slack"``: what a positive margin means for the model's module."""
    semantics = getattr(model.module, "margin_semantics", None)
    if semantics not in ("strict", "slack"):
        raise TypeError(
            f"{type(model.module).__name__} does not declare 'margin_semantics' "
            f"('strict' or 'slack'), so its margin cannot be set by meaning.")
    return semantics


class _InclusionMargin(TrainingRegime):
    _form = None

    def __init__(self, margin):
        super().__init__()
        if float(margin) < 0:
            raise ValueError("The margin must be non-negative; choose the regime "
                             "(slack or strict) for its meaning.")
        self.margin = float(margin)

    def attach(self, model):
        super().attach(model)
        value = self.margin if margin_semantics(model) == self._form else -self.margin
        name = getattr(model.module, "margin_attribute", "margin")
        setattr(model.module, name, value)
        for attribute in dict.fromkeys(("margin", name)):
            if hasattr(model, attribute):
                setattr(model, attribute, value)


class SlackMargin(_InclusionMargin):
    """Inclusions may be violated by up to ``margin`` in every dimension (see the module
    documentation). With ``margin > 0`` a zero loss does not make the embedding a model.

    :param margin: The slack :math:`\\gamma \\ge 0`.
    """

    key = "slack-inclusion-margin"
    _form = "slack"


class StrictMargin(_InclusionMargin):
    """Inclusions must hold with a gap of at least ``margin`` on every side (see the
    module documentation). Zero loss implies that the axiom holds.

    :param margin: The gap :math:`\\gamma \\ge 0`.
    """

    key = "strict-inclusion-margin"
    _form = "strict"
