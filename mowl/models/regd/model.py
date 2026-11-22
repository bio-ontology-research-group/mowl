from contextlib import contextmanager

import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.representation.library.ballfam_energy import contrastive_loss
from mowl.nn.el.regd.module import RegDELBEModule, RegDELEmModule


@contextmanager
def _scoring(module):
    """The module's raw-energy scoring context; a geometry that is not a RegD module (its
    parts were swapped) gets the raw energy of its ``boundary and depth`` measure, if it has
    one, and is scored as it is otherwise."""
    if hasattr(module, "scoring"):
        with module.scoring():
            yield
    elif hasattr(getattr(module, "measure", None), "raw"):
        previous = module.measure.raw
        module.measure.raw = True
        try:
            yield
        finally:
            module.measure.raw = previous
    else:
        yield


class _EnergyView(th.nn.Module):
    """Calls the module in scoring mode (raw RegD energies), for the evaluators."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, data, gci_name, neg=False):
        with _scoring(self.module):
            return self.module(data, gci_name, neg=neg)


class _RegDModel(EmbeddingELModel):
    """Training and scoring shared by :class:`RegDELBE` and :class:`RegDELEm`."""

    def _set_regd_options(self, num_neg, lamda, p, gamma1, gamma2, eps, depth_on_containment):
        if num_neg < 1:
            raise ValueError("num_neg must be at least 1.")
        self.num_neg = num_neg
        self.lamda = lamda
        self.p = p
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.eps = eps
        self.depth_on_containment = depth_on_containment
        # K negatives per positive: one block per entry of corrupt_column. Corrupting the
        # superclass / filler follows the authors' ontology code.
        self._DEFAULT_NEG_SAMPLING_CONFIG = {
            "gci0": {"index_pool": "classes", "corrupt_column": [1] * num_neg},
            "gci2": {"index_pool": "classes", "corrupt_column": [2] * num_neg},
            "object_property_assertion": {"index_pool": "individuals", "corrupt_column": 2},
        }

    def _after_init_module(self):
        self.evaluation_model = _EnergyView(self.module)

    def compute_loss(self, pos_scores, neg_scores=None):
        """For ``gci0`` and ``gci2``, the contrastive loss of Section 3.3 (Eq. 10) with the
        ``num_neg`` negatives of each positive; for every other normal form, the host model's
        loss."""
        if neg_scores is not None and self._contrastive():
            n = pos_scores.shape[0]
            negatives = neg_scores.reshape(-1, n).t()
            return contrastive_loss(pos_scores.reshape(-1), negatives, self.gamma1)
        return super().compute_loss(pos_scores, neg_scores)

    def _contrastive(self):
        """Whether the last loss was of a form with the contrastive loss."""
        forms = getattr(self.module, "contrastive_gcis", frozenset({"gci0", "gci2"}))
        return getattr(self.module, "last_form", None) in forms

    def eval_method(self, data):
        with _scoring(self.module):
            return super().eval_method(data)

    def score(self, axiom):
        """RegD energy of an axiom (lower: more likely entailed). It is negative when the
        child region lies inside the parent region and the depth term does not cancel that."""
        with _scoring(self.module):
            return super().score(axiom)


class RegDELBE(_RegDModel):
    r"""ELBE ([peng2020]_) with RegD ([yang2025regd]_): boxes with a depth and a boundary
    dissimilarity instead of the box inclusion energy.

    Concepts are boxes, roles are translations. RegD replaces the inclusion energy of
    :math:`C \sqsubseteq D`, :math:`C_1 \sqcap C_2 \sqsubseteq D`, :math:`C \sqsubseteq \exists
    R.D` and :math:`\exists R.C \sqsubseteq D` (this is the set of normal forms the authors'
    code replaces for ELBE). ``gci0`` and ``gci2`` are trained with the contrastive loss of
    Section 3.3 and ``num_neg`` negatives per positive; the others minimise
    :math:`\max\{E, \gamma_1\}` with ELBE's mean-squared-error loss. Bottom concepts and
    assertions keep ELBE's losses. See :class:`~mowl.nn.el.regd.RegDELBEModule`.

    The paper's ontology experiments (Section 5.2, 5.3; Tables 3 and 4) use
    ``embed_dim`` 5 and 10, 5000 epochs, both margins 0, :math:`\lambda \in \{0, 0.5, 1\}`
    and a learning rate from ``{0.0005, 0.005, 0.01}`` (the sweep of the authors' code; the
    appendix says ``{0.01, 0.005, 0.001, 0.0005}`` for the DAG experiments), :math:`p = 1` for
    boxes, and 10 negatives per positive.

    .. note::
        Deviations and ambiguities (paper text vs the authors' released code, which produced
        the published numbers). (1) Eq. 9 adds :math:`\lambda d_{dep}` to every positive
        pair; the code adds it only to pairs that are not yet contained
        (``depth_on_containment=False`` reproduces the code; this class follows Eq. 9).
        (2) Eq. 10 combines a clamped positive energy with a max-margin over the
        negatives' *boundary* dissimilarity; the code instead minimises a softmax
        cross-entropy over the full energies of one positive and its 10 negatives
        (:class:`~mowl.training.SoftmaxEnergyContrastiveLoss`), and only on the normal
        form of the task (``gci0`` for the inference task, ``gci2`` for the prediction
        task); the other normal forms keep the host model's mean-squared-error loss.
        (3) The negatives replace the superclass of ``gci0`` and the filler of ``gci2``
        (the code's choice); the text says "the child". (4) For the prediction task the
        code's ``nf3Loss_ours`` constrains the *reverse* inclusion (see ``flip_gci2``).
        (5) The paper does not state batch size or initialisation; the code draws 512
        axioms per normal form per step, initialises uniformly in [-1, 1] and renormalises
        the rows (as mOWL's ELBE and ELEm modules do), and selects the checkpoint by the
        best F1 on the validation set (:class:`~mowl.training.ValidationF1Selection`).
        (6) In the paper's text the box host is called ELEM and the ball host ELBE in
        Section 5.2; the tables and the code use ELBE for boxes.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of the boxes.
    :param margin: Margin of the host model's losses (not of RegD).
    :param lamda: Weight :math:`\lambda` of the depth dissimilarity.
    :param p: Exponent of the :math:`L_p` norm in the depth dissimilarity.
    :param gamma1: Margin :math:`\gamma_1` of the positive loss term.
    :param gamma2: Margin :math:`\gamma_2` of the negative loss term.
    :param num_neg: Negatives per positive axiom of ``gci0`` and ``gci2``.
    :param eps: Lower bound of the offsets.
    :param depth_on_containment: Whether the depth term is also added to contained pairs.
    :param flip_gci2: Train the *reverse* inclusion for ``gci2`` (box(D) + R inside
        box(C) instead of box(C) + R inside box(D)). This is what the authors'
        prediction-task code does (``nf3Loss_ours``); its evaluation then scores
        box(D) − R inside the candidates (a different sign of R than in training).
        Defaults to ``False`` (the normal form's own direction).
    """

    def __init__(self,
                 dataset,
                 embed_dim=5,
                 margin=0,
                 lamda=1.0,
                 p=1,
                 gamma1=0.0,
                 gamma2=0.0,
                 num_neg=10,
                 eps=1e-4,
                 depth_on_containment=True,
                 flip_gci2=False,
                 learning_rate=0.005,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis)
        self.margin = margin
        self._loaded = False
        self.extended = False
        self._set_regd_options(num_neg, lamda, p, gamma1, gamma2, eps, depth_on_containment)
        self.flip_gci2 = bool(flip_gci2)
        self.init_module()

    def init_module(self):
        self.module = RegDELBEModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim,
            margin=self.margin,
            lamda=self.lamda, p=self.p, gamma1=self.gamma1, gamma2=self.gamma2,
            eps=self.eps, depth_on_containment=self.depth_on_containment,
            flip_gci2=self.flip_gci2,
        ).to(self.device)
        self._after_init_module()

    def compute_loss(self, pos_scores, neg_scores=None):
        """See :meth:`_RegDModel.compute_loss`. Other normal forms: ELBE's mean-squared-error
        loss (positives towards 0, negatives towards 1)."""
        if neg_scores is not None and self._contrastive():
            return super().compute_loss(pos_scores, neg_scores)
        pos_mean = th.mean(pos_scores)
        loss = th.square(pos_mean)
        if neg_scores is not None:
            loss = loss + th.square(th.mean(neg_scores) - 1.0)
        return loss


class RegDELEm(_RegDModel):
    r"""ELEm ([kulmanov2019]_) with RegD ([yang2025regd]_): balls with a depth and a boundary
    dissimilarity instead of the ball inclusion energy.

    Concepts are balls, roles are translations. RegD replaces the inclusion energy of
    :math:`C \sqsubseteq D`, :math:`C \sqsubseteq \exists R.D` and :math:`\exists R.C
    \sqsubseteq D`. The intersection form, bottom concepts, assertions and the centre-norm
    regularisation keep ELEm's. ``gci0`` and ``gci2`` use the contrastive loss of Section 3.3
    with ``num_neg`` negatives per positive. See :class:`~mowl.nn.el.regd.RegDELEmModule`.

    Hyperparameters of the paper's ontology experiments and the deviations from the paper:
    see :class:`RegDELBE`. For balls the paper finds :math:`p = 2` best, which is the default
    here.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of the balls.
    :param margin: Margin of the host model's losses (not of RegD).
    :param reg_norm: Target norm of the centres in ELEm's regularisation.
    :param lamda: Weight :math:`\lambda` of the depth dissimilarity.
    :param p: Exponent of the :math:`L_p` norm in the depth dissimilarity.
    :param gamma1: Margin :math:`\gamma_1` of the positive loss term.
    :param gamma2: Margin :math:`\gamma_2` of the negative loss term.
    :param num_neg: Negatives per positive axiom of ``gci0`` and ``gci2``.
    :param eps: Lower bound of the radii.
    :param depth_on_containment: Whether the depth term is also added to contained pairs.
    :param flip_gci2: See :class:`RegDELBE`.
    :param radius: ``"abs"`` (default) or ``"softplus"`` (the RegD code's ``Elem``); see
        :class:`~mowl.nn.el.regd.RegDELEmModule`.
    :param consistent_radius: See :class:`~mowl.nn.el.regd.RegDELEmModule`. Defaults to
        ``True``.
    :param gci1_radius_term: See :class:`~mowl.models.ELEmbeddings`.
    """

    def __init__(self,
                 dataset,
                 embed_dim=5,
                 margin=0,
                 reg_norm=1,
                 lamda=1.0,
                 p=2,
                 gamma1=0.0,
                 gamma2=0.0,
                 num_neg=10,
                 eps=1e-4,
                 depth_on_containment=True,
                 flip_gci2=False,
                 radius="abs",
                 gci1_radius_term=True,
                 consistent_radius=True,
                 learning_rate=0.005,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis)
        self.margin = margin
        self.reg_norm = reg_norm
        self._loaded = False
        self.extended = False
        self._set_regd_options(num_neg, lamda, p, gamma1, gamma2, eps, depth_on_containment)
        self.flip_gci2 = bool(flip_gci2)
        self.radius = radius
        self.gci1_radius_term = gci1_radius_term
        self.consistent_radius = consistent_radius
        self.init_module()

    def init_module(self):
        self.module = RegDELEmModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim,
            margin=self.margin,
            reg_norm=self.reg_norm,
            lamda=self.lamda, p=self.p, gamma1=self.gamma1, gamma2=self.gamma2,
            eps=self.eps, depth_on_containment=self.depth_on_containment,
            flip_gci2=self.flip_gci2, radius=self.radius,
            gci1_radius_term=self.gci1_radius_term,
            consistent_radius=self.consistent_radius,
        ).to(self.device)
        self._after_init_module()
