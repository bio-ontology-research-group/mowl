from mowl.baselines.base_models.elmodel import EmbeddingELModel
from mowl.baselines.nn.el.transbox.module import TransBoxModule


class TransBox(EmbeddingELModel):
    r"""
    Implementation based on [yang2025]_.

    TransBox is an EL++-closed ontology embedding method: atomic concepts and
    roles are embedded as axis-aligned boxes (center + non-negative offset)
    and individuals as points. Complex EL++ concepts are embedded by
    composing the atomic boxes, so the model can score axioms involving
    complex concepts (e.g. ``C ⊑ A`` with a complex ``C``) by construction.

    .. note::
        The paper's benchmarks use ``embed_dim=200``, ``learning_rate=0.0005``,
        ``margin=0`` and ``reg_factor=0`` for GALEN, GO and Anatomy.

    .. note::
        Following the paper (Section 4.1), GCIs of the form ``C \sqsubseteq \exists R.D``
        are negated on **both sides**: the negative sampling config corrupts the ``C``
        and the ``D`` columns separately (two negatives per positive), and the module's
        gci2 negative loss implements the paper's :math:`L_{\nsubseteq}`.

    .. note::
        The module implements the EL++ role axiom losses and declares
        ``role_axiom_capable``, so ``TransBox`` accepts ``load_role_axioms=True`` to
        train on role inclusions and chains alongside the concept normal forms.

    .. note::
        ``formulation="reference"`` switches to the loss functions and the
        initialisation of the authors' code (see :class:`~mowl.nn.TransBoxModule` and
        :mod:`mowl.baselines.nn.el.transbox.code_losses`). Train it with the regime
        :class:`~mowl.training.TransBoxCodeTraining`, which squares the losses per axiom
        and pushes negatives to a target; the default ``"mowl"`` formulation is
        unchanged. ``strict_negatives`` and ``reference_bottom_losses`` select the
        earlier behaviour of two details, see :class:`~mowl.nn.TransBoxModule`.

    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0.1,
                 learning_rate=0.001,
                 batch_size=4096,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None,
                 use_enhancement=True,
                 reg_factor=0.1,
                 load_role_axioms=False,
                 formulation="mowl",
                 strict_negatives=True,
                 reference_bottom_losses="code"
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis,
                         load_role_axioms=load_role_axioms)

        # Paper-faithful negative sampling: negate both sides of C ⊑ ∃R.D
        # (corrupt C and D separately instead of D only).
        self._DEFAULT_NEG_SAMPLING_CONFIG = {
            "gci2": {"index_pool": "classes", "corrupt_column": [0, 2]},
        }

        self.margin = margin
        self.use_enhancement = use_enhancement
        self.reg_factor = reg_factor
        self.formulation = formulation
        self.strict_negatives = strict_negatives
        self.reference_bottom_losses = reference_bottom_losses
        self.init_module()

    def init_module(self):
        self.module = TransBoxModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim,
            margin=self.margin,
            use_enhancement=self.use_enhancement,
            reg_factor=self.reg_factor,
            formulation=self.formulation,
            strict_negatives=self.strict_negatives,
            reference_bottom_losses=self.reference_bottom_losses,
        ).to(self.device)
