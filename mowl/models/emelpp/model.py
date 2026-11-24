from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.el.emelpp.module import EmELPPModule, EmELVarModule


class EmELPP(EmbeddingELModel):
    r"""
    Implementation based on [mondal2021]_.

    EmEL++ extends ELEmbeddings ([kulmanov2019]_): classes are :math:`n`-balls and roles are
    translations, and in addition there are losses for role inclusions
    :math:`R \sqsubseteq S` and role chains :math:`R_1 \circ R_2 \sqsubseteq S`. Train with
    ``load_role_axioms=True`` to use them. Individuals are treated as nominal classes.

    The paper trains with Adam on the sum of the losses of Eq. 1-10, with negatives
    generated for :math:`C \sqsubseteq \exists R.D` by replacing :math:`C` or :math:`D` by a
    random class (Section 3.3); the model does the same by default (both columns, one
    negative each).

    .. note::
        Hyperparameters of the paper (Table 2, best on the validation set, learning rate and
        batch size are not reported): GALEN ``embed_dim=50, margin=0.0``; GO
        ``embed_dim=100, margin=-0.1``; Anatomy ``embed_dim=200, margin=-0.1``; SNOMED CT
        ``embed_dim=100, margin=-0.1``. The grid was :math:`n \in \{50, 100, 200\}` and
        :math:`\gamma \in \{-0.1, 0, 0.1\}`.

    .. note::
        The role inclusion loss of the paper is symmetric in the two roles, so
        :math:`R \sqsubseteq S` also enforces :math:`S \sqsubseteq R`. This is implemented
        as published; see the catalogue card.

    .. note::
        Differences to the paper: radii are used through their absolute value (the paper
        requires them to be non-negative and mentions a penalty for negative radii without
        giving it); the normalisation terms of Eq. 1-10 are part of each axiom loss, so
        there is no extra regularisation term.

    :param dataset: mOWL dataset.
    :param embed_dim: Embedding dimension :math:`n`.
    :param margin: Margin :math:`\gamma`.
    :param learning_rate: Adam learning rate.
    :param batch_size: Batch size.
    :param model_filepath: Where the best model is stored.
    :param device: ``'cpu'`` or ``'cuda'``.
    :param neg_sampling_gcis: GCIs that use negative sampling. ``None`` selects all that the
        module supports (``gci2`` and ``object_property_assertion``).
    :param load_role_axioms: Load role inclusions and role chains and train on them.
    """

    _MODULE_CLASS = EmELPPModule

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0.1,
                 learning_rate=0.001,
                 batch_size=4096,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None,
                 load_role_axioms=False):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis,
                         load_role_axioms=load_role_axioms)

        # Section 3.3: C ⊑ ∃R.D is corrupted in C or in D.
        self._DEFAULT_NEG_SAMPLING_CONFIG = dict(
            self._DEFAULT_NEG_SAMPLING_CONFIG,
            gci2={"index_pool": "classes", "corrupt_column": [0, 2]})

        self.margin = margin
        self.init_module()

    def _module_kwargs(self):
        return {}

    def init_module(self):
        self.module = self._MODULE_CLASS(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin,
            **self._module_kwargs()
        ).to(self.device)


class EmELVar(EmELPP):
    r"""
    Implementation based on [mohapatra2021]_ (summarised in [mohapatra2021a]_).

    EmEL(var) is EmEL++ with a learned variance :math:`\sigma(r) \ge 0` per role, which lets
    the translated ball :math:`C + R` lie within :math:`\sigma(R)` of :math:`D`, so that a
    role can relate a class to several, far apart, fillers (one-to-many and many-to-many
    roles). :math:`\sigma` enters the losses of :math:`C \sqsubseteq \exists R.D` and
    :math:`\exists R.C \sqsubseteq D` only (Eq. 7 and 8 of the paper).

    .. note::
        The paper does not report hyperparameters for EmEL(var) beyond using the setup of
        EmEL++; use the values listed for :class:`EmELPP`. It reports results on GALEN, GO
        and SNOMED CT (Tables 1-3), without individuals.

    .. note::
        As published, the :math:`+\sigma(r)` term of Eq. 7 and 8 has weight 1 in every
        axiom, which cancels the effect of the variance on the hinge (see
        :class:`~mowl.nn.el.emelpp.module.EmELVarModule`). ``sigma_reg_weight`` below is an
        option that is **not part of the paper**; the default (1) is the paper's loss.

    :param sigma_init: Initial :math:`\sigma` (not given in the paper).
    :param sigma_reg_weight: Weight of the term :math:`+\sigma(r)` in Eq. 7 and 8. The paper
        uses 1.

    The remaining parameters are those of :class:`EmELPP`.
    """

    _MODULE_CLASS = EmELVarModule

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0.1,
                 learning_rate=0.001,
                 batch_size=4096,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None,
                 load_role_axioms=False,
                 sigma_init=0.1,
                 sigma_reg_weight=1.0):
        self.sigma_init = sigma_init
        self.sigma_reg_weight = sigma_reg_weight
        super().__init__(dataset, embed_dim=embed_dim, margin=margin,
                         learning_rate=learning_rate, batch_size=batch_size,
                         model_filepath=model_filepath, device=device,
                         neg_sampling_gcis=neg_sampling_gcis,
                         load_role_axioms=load_role_axioms)

    def _module_kwargs(self):
        return {"sigma_init": self.sigma_init, "sigma_reg_weight": self.sigma_reg_weight}
