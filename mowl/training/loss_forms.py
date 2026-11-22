"""Variants of the loss terms of a model, as training regimes.

Re-implementations of a method often differ from its reference code in small details of
the loss: whether a negative loss has a hinge, where the normalisation penalty of the
centres is applied, whether the batch losses are averaged or summed, which regulariser is
used. These details are training choices, not part of the representation, so mOWL's
modules keep their default and expose the alternative as an attribute that one of the
regimes here sets.

* :class:`UnboundedNegativeLoss`: the negative loss of :math:`C \\sqsubseteq \\exists R.D`
  without the hinge (ELEm, EmEL++).
* :class:`PerAxiomNormPenalty`: the unit-norm penalty of the class centres in every axiom
  loss instead of once per step (ELEm).
* :class:`Box2ELBaselineLosses`: the loss details of the baseline implementations of the
  Box\\ :sup:`2`\\ EL code (``model/Elem.py``, ``model/EmELpp.py``, ``model/BoxEL.py``).
"""

from mowl.training.regime import TrainingRegime


def _is(module, name):
    return any(cls.__name__ == name for cls in type(module).__mro__)


def _set(model, attribute, value, families):
    module = model.module
    if not any(_is(module, f) for f in families):
        raise TypeError(f"{type(module).__name__} has no option '{attribute}' "
                        f"(supported: {', '.join(families)} and subclasses).")
    setattr(module, attribute, value)


class UnboundedNegativeLoss(TrainingRegime):
    """The negative loss of :math:`C \\not\\sqsubseteq \\exists R.D` without the hinge:
    :math:`r_C + r_D - \\lVert c + R - d \\rVert + \\gamma` instead of
    :math:`\\max(0, \\cdot)`. The loss keeps falling as the two balls move apart and as
    their radii shrink, so in combination with the unit-norm penalty it pushes all radii
    towards zero. This is the negative loss of the ELEm and EmEL++ baselines of the
    Box\\ :sup:`2`\\ EL code (``neg_loss``); ELEmbeddings and EmEL++ (Eq. 10) state it
    with the hinge.
    """

    key = "unbounded-negative-loss"

    def attach(self, model):
        super().attach(model)
        _set(model, "negative_loss_form", "linear", ("ELEmModule",))


class PerAxiomNormPenalty(TrainingRegime):
    """The penalty :math:`|\\lVert c \\rVert - 1|` of every class centre that occurs in an
    axiom is added to the loss of that axiom (positives and negatives alike), instead of
    once per step as the mean over all classes. Classes that occur in many axioms are
    penalised more often. This is the regularisation of the original ELEmbeddings code and
    of the ELEm baseline of the Box\\ :sup:`2`\\ EL code (``reg`` in every ``nf*Loss``);
    EmEL++ always does this (its normalisation terms are part of Eq. 1-10).
    """

    key = "per-axiom-norm-penalty"

    def attach(self, model):
        super().attach(model)
        if _is(model.module, "EmELPPModule"):
            return
        _set(model, "norm_penalty", "per_axiom", ("ELEmModule",))


class Box2ELBaselineLosses(TrainingRegime):
    """The loss details of the baseline models of the Box\\ :sup:`2`\\ EL code
    (KRR-Oxford/BoxSquaredEL, ``model/Elem.py``, ``model/EmELpp.py``, ``model/BoxEL.py``)
    that differ from mOWL's modules:

    * ELEm and EmEL++: the negative loss has no hinge (:class:`UnboundedNegativeLoss`);
    * ELEm: the unit-norm penalty is part of every axiom loss
      (:class:`PerAxiomNormPenalty`);
    * EmEL++: the loss of :math:`C \\sqcap D \\sqsubseteq E` has no term on the radius of
      :math:`E` (it is commented out in the code);
    * BoxEL: the losses of a batch are *summed* (not averaged) and the regulariser is the
      code's ``l2_side_regularizer`` of every box of every batch
      (:func:`mowl.nn.el.boxel.losses.box2el_code_side_regularizer`).

    Other modules (Box2EL, ELBE) are left unchanged.

    :param negative_loss: Apply the unbounded negative loss (ELEm, EmEL++).
    :param norm_penalty: Apply the per-axiom unit-norm penalty (ELEm).
    :param radius_term: Keep EmEL++'s radius term (``False`` as in the code).
    :param boxel: Apply the BoxEL summation and regulariser.
    """

    key = "box2el-baseline-losses"

    def __init__(self, negative_loss=True, norm_penalty=True, radius_term=False, boxel=True):
        super().__init__()
        self.negative_loss = bool(negative_loss)
        self.norm_penalty = bool(norm_penalty)
        self.radius_term = bool(radius_term)
        self.boxel = bool(boxel)
        self._sum = False

    def attach(self, model):
        super().attach(model)
        module = model.module
        self._sum = False
        if _is(module, "ELEmModule"):
            if self.negative_loss:
                UnboundedNegativeLoss().attach(model)
            if self.norm_penalty:
                PerAxiomNormPenalty().attach(model)
            if _is(module, "EmELPPModule"):
                module.intersection_radius_term = self.radius_term
        elif _is(module, "BoxELModule") and self.boxel:
            module.regulariser = "box2el_code"
            module.positive_scale = False  # the code's signed role scale
            self._sum = True

    def loss(self, gci_name, pos_scores, neg_scores):
        if not self._sum:
            return NotImplemented
        loss = pos_scores.sum()
        if neg_scores is not None:
            loss = loss + neg_scores.sum()
        return loss


_THING = "http://www.w3.org/2002/07/owl#Thing"


class EmELppCodeObjective(TrainingRegime):
    """The objective of the Keras code of EmEL++ (kracr/EmELpp, ``*_EmEL.py``) and of the
    ELEmbeddings baseline in the same repository (``*_EL.py``), which differs from the
    losses printed in the paper:

    * a step draws ``batch_size`` rows of every normal form; row :math:`i` of all normal
      forms (positives, negatives, role axioms) is summed into one number :math:`s_i`,
      and the model is fitted with Keras' ``mse`` loss against zero, i.e. the objective is
      :math:`\\mathrm{mean}_i\\, s_i^2`. Since the negative loss has no hinge
      (:class:`UnboundedNegativeLoss`) and can be negative, the squared sum drives
      :math:`s_i` to zero rather than every term to its minimum;
    * :math:`C \\sqsubseteq D` is trained as :math:`C \\sqsubseteq \\exists r_{sub}.D`
      with a learned "SubClassOf" translation :math:`r_{sub}` (testing uses the plain
      centre distance);
    * every row contains :math:`|r_\\top - 5|`, which pulls the radius of
      ``owl:Thing`` to 5, and (EmEL++ only) :math:`\\max(0, -r_X)` for a random class
      :math:`X`;
    * EmEL++ uses :math:`\\max(0, r)` as radius (``radius_form="relu"``), ELEm
      :math:`|r|`; the unit-norm penalties are part of every axiom loss.

    Row sums need every normal form to contribute exactly ``batch_size`` rows; use it
    with :class:`~mowl.training.PerNormalFormMiniBatches` (``sample_small=True``) and
    :class:`~mowl.training.FixedNegatives` of the same batch size. Rows of a normal form
    with a different number of rows are added as their mean.

    :param batch_size: Rows per normal form and step (256 in the code).
    :param top_radius: Target radius of ``owl:Thing`` (``self.inf = 5.0``).
    :param subclass_translation: Train :math:`C \\sqsubseteq D` with the translation.
    :param batch_direction: Compute the direction term of the EmEL++ role axiom losses
        as the code does (``tf.nn.l2_normalize`` along the batch axis: one number per
        batch, added to every row; a bug of the code, kept here because the published
        numbers were obtained with it). ``False`` uses the per-row cosine of the paper.
    :param seed: Seed of the class draws of the radius term and of the initial
        translation.
    """

    key = "emelpp-code-objective"

    def __init__(self, batch_size=256, top_radius=5.0, subclass_translation=True,
                 batch_direction=True, seed=None):
        super().__init__()
        self.batch_direction = bool(batch_direction)
        self.batch_size = int(batch_size)
        self.top_radius = float(top_radius)
        self.subclass_translation = bool(subclass_translation)
        self.seed = seed
        self._rows = []

    def attach(self, model):
        import torch as th
        super().attach(model)
        module = model.module
        if not _is(module, "ELEmModule"):
            raise TypeError(f"{type(module).__name__} is not an ELEm/EmEL++ module.")
        self._emelpp = _is(module, "EmELPPModule")
        module.negative_loss_form = "linear"
        if self._emelpp:
            module.radius_form = "relu"
            if self.batch_direction:
                module.role_direction = "batch"
        else:
            module.norm_penalty = "per_axiom"
        gen = th.Generator(device="cpu")
        gen.manual_seed(0 if self.seed is None else int(self.seed))
        self._gen = gen
        if self.subclass_translation and getattr(module, "subclass_relation", None) is None:
            dim = module.class_embed.weight.shape[1]
            v = th.rand(dim, generator=gen) * 2 - 1
            v = v / th.linalg.norm(v)
            module.register_parameter(
                "subclass_relation",
                th.nn.Parameter(v.to(module.class_embed.weight.device)))
        self._top = getattr(model, "class_index_dict", {}).get(_THING)
        self._rows = []

    def _radius(self, idx):
        rad = self.model.module.class_rad(idx)
        import torch as th
        return th.relu(rad) if self._emelpp else th.abs(rad)

    def loss(self, gci_name, pos_scores, neg_scores):
        self._rows.append(pos_scores)
        if neg_scores is not None:
            self._rows.append(neg_scores)
        return pos_scores.sum() * 0.0

    def additional_loss(self, epoch):
        import torch as th
        rows, self._rows = self._rows, []
        if not rows:
            return None
        device = rows[0].device
        b = self.batch_size
        total = th.zeros(b, 1, device=device)
        for r in rows:
            r = r.reshape(len(r), -1).sum(dim=1, keepdim=True)
            total = total + (r if len(r) == b else r.mean())
        weight = self.model.module.class_rad.weight
        if self._top is not None:
            top = self._radius(th.tensor([self._top], device=weight.device))
            total = total + th.abs(top.reshape(1, 1) - self.top_radius)
        if self._emelpp:
            idx = th.randint(0, weight.shape[0], (b,), generator=self._gen).to(weight.device)
            total = total + th.relu(-self.model.module.class_rad(idx)).reshape(b, 1)
        return (total ** 2).mean()
