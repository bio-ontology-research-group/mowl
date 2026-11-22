from mowl.evaluation import SubsumptionEvaluator
from mowl.representation.library.logic_alc import AnonymousElements


class FALCONSubsumptionEvaluator(SubsumptionEvaluator):
    """Subsumption ranking evaluator for :class:`FALCONModel \
    <mowl.models.falcon.model.FALCONModel>`.

    Scores a subsumption :math:`C \\sqsubseteq D` by the FALCON *concept-concept* energy
    :math:`-\\log(1 - \\max_e \\mu(C \\sqcap \\neg D, e))`, evaluated over a sample of
    (named and anonymous) entities: the ``inside`` of the measure of the geometry
    (:class:`~mowl.representation.library.logic_fuzzy.ConflictMeasure`) on a context of the
    named entities and ``n_anon`` fresh anonymous ones
    (:class:`~mowl.representation.library.logic_alc.AnonymousElements` without
    perturbations). Lower energy means the subsumption is more strongly satisfied, which is
    the convention expected by the ranking framework.

    :param n_anon: Number of anonymous entities sampled to form the membership domain. \
    Defaults to ``16``.
    :type n_anon: int, optional
    """

    def __init__(self, *args, n_anon=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_anon = n_anon

    def _entity_context(self, module):
        sampler = AnonymousElements(self.n_anon, perturbed=False)
        named = module.e_embedding.weight
        sample = sampler.sample(named.detach(), named.shape[1], named.device)
        return sampler.context(named, named.shape[0], sample)

    def get_scores(self, model, batch):
        return model.class_inclusion_costs(batch[:, 0], batch[:, 1],
                                           self._entity_context(model))
