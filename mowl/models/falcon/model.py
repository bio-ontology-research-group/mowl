from mowl.models.composed.alc import ComposedALC
from mowl.nn.alc.falcon.module import FALCONModule
from mowl.base_models.alcmodel import EmbeddingALCModel  # noqa: F401  (a name of 2.1.0)

_THING = "http://www.w3.org/2002/07/owl#Thing"
_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


class FALCONModel(ComposedALC):
    """FALCON model [falcon2022]_: a fuzzy :math:`\\mathcal{ALC}` neural reasoner.

    Each class expression is interpreted as a fuzzy set over a collection of named and
    sampled anonymous entities, and axioms are scored through fuzzy logical operators.
    The training objective combines a TBox term and two ABox terms (concept and relation
    assertions), weighted by ``alpha`` and ``beta`` (see
    :class:`EmbeddingALCModel <mowl.base_models.alcmodel.EmbeddingALCModel>`).

    :param dataset: mOWL dataset to use for training.
    :type dataset: :class:`mowl.datasets.base.Dataset`
    :param embed_dim: Dimension of the embeddings. Defaults to ``50``.
    :type embed_dim: int, optional
    :param anon_e: Number of anonymous entities sampled each epoch. Defaults to ``4``.
    :type anon_e: int, optional
    :param t_norm: Fuzzy t-norm: ``'product'``, ``'minmax'`` or ``'Łukasiewicz'``. \
    Defaults to ``'product'``.
    :type t_norm: str, optional
    :param max_measure: Aggregation used by the concept-concept loss. Defaults to ``'max'``.
    :type max_measure: str, optional
    :param num_negs: Number of negative samples per ABox axiom. Defaults to ``4``.
    :type num_negs: int, optional
    :param role_pairs: ``'all'`` (default, the original code): :math:`\\exists r.C` and
        :math:`\\forall r.C` are evaluated over all pairs of entities, one degree per entity.
        ``'self'``: only pairs of an entity with itself (one value for every entity; the
        behaviour of earlier mOWL versions, a porting error). See
        :class:`~mowl.baselines.nn.alc.falcon.module.FALCONModule`.
    :type role_pairs: str, optional
    :param thing: ``'learned'`` (default, the original code) or ``'all'``: ``owl:Thing`` is
        the constant set with degree 1.
    :type thing: str, optional
    :param nothing: ``'empty'`` (default, the original code): ``owl:Nothing`` is the constant
        empty set; ``'learned'``: an ordinary learned class (earlier mOWL versions).
    :type nothing: str, optional

    The model is a combination of parts (:class:`~mowl.models.ComposedALC` with the method
    ``"falcon"``, see :func:`mowl.representation.library.logic_fuzzy.falcon`): its module
    (:class:`~mowl.nn.alc.falcon.module.FALCONModule`) is a
    :class:`~mowl.representation.Geometry` whose class model is
    :class:`~mowl.representation.library.logic_fuzzy.NetworkClasses`, relation model
    :class:`~mowl.representation.library.logic_fuzzy.PairNetworkRelations`, measure
    :class:`~mowl.representation.library.logic_fuzzy.ConflictMeasure` and objective
    :class:`~mowl.representation.library.logic_fuzzy.FuzzyObjective` (assertions contrasted with
    sampled corrupted ones). The hand-written implementation it replaces is
    :class:`mowl.baselines.models.falcon.model.FALCONModel`.
    """

    def __init__(self, dataset, embed_dim=50, anon_e=4, alpha=0.3, beta=0.3,
                 t_norm='product', max_measure='max', num_negs=4,
                 learning_rate=0.001, batch_size=256, model_filepath=None, device='cpu',
                 role_pairs='all', thing='learned', nothing='empty'):
        self.t_norm = t_norm
        self.max_measure = max_measure
        self.num_negs = num_negs
        self.role_pairs = role_pairs
        self.thing = thing
        self.nothing = nothing
        super().__init__(dataset, method="falcon", embed_dim=embed_dim,
                         batch_size=batch_size, alpha=alpha, beta=beta, anon_e=anon_e,
                         learning_rate=learning_rate, model_filepath=model_filepath,
                         device=device)

    def init_module(self):
        heads_dict, tails_dict = self._build_assertion_dicts()
        # The membership network always needs at least one named entity row, even for
        # purely terminological (TBox-only) ontologies with no individuals.
        nentities = max(len(self.dataset.individuals), 1)
        self.module = FALCONModule(
            len(self.dataset.classes),
            nentities,
            len(self.dataset.object_properties),
            heads_dict,
            tails_dict,
            embed_dim=self.embed_dim,
            anon_e=self.anon_e,
            t_norm=self.t_norm,
            max_measure=self.max_measure,
            num_negs=self.num_negs,
            device=self.device,
            role_pairs=self.role_pairs,
            top_index=self.class_index_dict.get(_THING),
            bottom_index=self.class_index_dict.get(_NOTHING),
            thing=self.thing,
            nothing=self.nothing,
        ).to(self.device)
