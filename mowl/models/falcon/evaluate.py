import torch as th
from mowl.evaluation import SubsumptionEvaluator


class FALCONSubsumptionEvaluator(SubsumptionEvaluator):
    """Subsumption ranking evaluator for :class:`FALCONModel \
    <mowl.models.falcon.model.FALCONModel>`.

    Scores a subsumption :math:`C \\sqsubseteq D` by the FALCON *concept-concept* energy
    :math:`-\\log(1 - \\max_e \\mu(C \\sqcap \\neg D, e))`, evaluated over a sample of
    (named and anonymous) entities. Lower energy means the subsumption is more strongly
    satisfied, which is the convention expected by the ranking framework.

    :param n_anon: Number of anonymous entities sampled to form the membership domain. \
    Defaults to ``16``.
    :type n_anon: int, optional
    """

    def __init__(self, *args, n_anon=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_anon = n_anon

    def _entity_context(self, module):
        named = module.e_embedding.weight
        dim = named.shape[1]
        anon = th.empty(self.n_anon, dim, device=named.device)
        th.nn.init.xavier_uniform_(anon)
        if named.shape[0] > 0:
            return th.cat([named, anon], dim=0)
        return anon

    def get_scores(self, model, batch):
        c, d = batch[:, 0], batch[:, 1]
        e_emb = self._entity_context(model)
        c_fs = model._get_c_fs_batch(model.c_embedding(c), e_emb)
        d_fs = model._get_c_fs_batch(model.c_embedding(d), e_emb)
        # Fuzzy membership of C ⊓ ¬D; if it is satisfiable (max > 0) the subsumption
        # is violated, so a higher value yields a higher (worse) energy/score.
        cc = model._logical_and(c_fs, model._logical_not(d_fs))
        scores = - th.log(1 - cc.max(dim=-1)[0] + 1e-10)
        return scores
