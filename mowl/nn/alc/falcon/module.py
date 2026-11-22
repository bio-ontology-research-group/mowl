from types import SimpleNamespace

import torch.nn as nn

from mowl.representation.library.logic_alc import ExpressionGeometry
from mowl.representation.library.logic_fuzzy import falcon_parts

__all__ = ["FALCONModule"]


class FALCONModule(ExpressionGeometry):
    """The FALCON model [falcon2022]_, a fuzzy :math:`\\mathcal{ALC}` neural reasoner, put
    together from parts. Each class expression is mapped to a *fuzzy set* over a collection
    of (named and anonymous) entity embeddings, and axioms are scored through fuzzy logical
    operators.

    The module is the geometry of the method
    :func:`~mowl.representation.library.logic_fuzzy.falcon` (class model
    :class:`~mowl.representation.library.logic_fuzzy.NetworkClasses`, relation model
    :class:`~mowl.representation.library.logic_fuzzy.PairNetworkRelations`, measure
    :class:`~mowl.representation.library.logic_fuzzy.ConflictMeasure`, objective
    :class:`~mowl.representation.library.logic_fuzzy.FuzzyObjective`); the hand-written loss
    module it replaces is kept for comparison as :mod:`mowl.baselines.nn.alc.falcon`.
    Based on the original implementation at
    https://github.com/bio-ontology-research-group/FALCON

    :param nclasses: Number of classes.
    :param nentities: Number of named entities (individuals); at least ``1``.
    :param nrelations: Number of object properties.
    :param heads_dict: ``(tail, role) -> heads`` of the asserted role assertions, to draw
        corrupted ones from.
    :param tails_dict: ``(head, role) -> tails``.
    :param role_pairs: How :math:`\\exists r.C` and :math:`\\forall r.C` read the role.
        ``"all"`` (default, as in the original code): the degree of an entity :math:`x` is
        taken over all pairs :math:`(x, y)` of the domain, :math:`\\max_y r(x, y) \\sqcap C(y)`
        (and :math:`\\min_y` of the residuum for :math:`\\forall`), with
        :math:`r(x, y) = \\mu(e_x + r, e_y)`; this needs memory quadratic in the domain.
        ``"self"``: the pair of an entity with itself only, :math:`r(x, x)`, so one value
        is computed and given to every entity (a porting error of earlier mOWL versions, not
        the published method).
    :param top_index: Class index of ``owl:Thing`` (or ``None``).
    :param bottom_index: Class index of ``owl:Nothing`` (or ``None``).
    :param thing: ``"learned"`` (default, as in the original code): ``owl:Thing`` has an
        embedding like any other class. ``"all"``: it is the constant fuzzy set with degree 1.
    :param nothing: ``"empty"`` (default, as in the original code): ``owl:Nothing`` is the
        constant empty fuzzy set. ``"learned"``: an ordinary learned class.
    :param loss_type: ``"c"`` (default): a sampled role assertion contrasts its degree with
        those of the corrupted ones (``"contrast"`` of the objective); ``"r"``: a ranking
        loss between them.
    """

    #: The anonymous entities are sampled every epoch by the training loop.
    samples_anonymous = True

    def __init__(self, nclasses, nentities, nrelations, heads_dict, tails_dict, embed_dim=128,
                 anon_e=4, t_norm='product', max_measure='max', residuum='notCorD',
                 loss_type='c', num_negs=4, device='cpu', role_pairs='all', top_index=None,
                 bottom_index=None, thing='learned', nothing='empty'):
        if role_pairs not in ('all', 'self'):
            raise ValueError("role_pairs must be 'all' or 'self'.")
        if residuum != 'notCorD':
            raise ValueError("residuum must be 'notCorD'.")
        if loss_type not in ('c', 'r'):
            raise ValueError("loss_type must be 'c' or 'r'.")
        # The tables are drawn in the order of the earlier module (embeddings, then the two
        # layers, then the Xavier initialisation of all), so that a seed gives the numbers it
        # always gave.
        c_embedding = nn.Embedding(nclasses, embed_dim)
        r_embedding = nn.Embedding(nrelations, embed_dim)
        e_embedding = nn.Embedding(nentities, embed_dim)
        fc_0 = nn.Linear(embed_dim * 2, embed_dim)
        fc_1 = nn.Linear(embed_dim, 1)
        for table in (c_embedding, r_embedding, e_embedding, fc_0, fc_1):
            nn.init.xavier_uniform_(table.weight.data)
        tables = SimpleNamespace(
            c_embedding=c_embedding, r_embedding=r_embedding, e_embedding=e_embedding,
            fc_0=fc_0, fc_1=fc_1, t_norm=t_norm, max_measure=max_measure, role_pairs=role_pairs,
            top_index=top_index if thing == 'all' else None,
            bottom_index=bottom_index if nothing == 'empty' else None,
            num_negs=num_negs, heads_dict=heads_dict, tails_dict=tails_dict,
            nentities=nentities)
        parts = falcon_parts(None, None, t_norm=t_norm, thing=thing, nothing=nothing,
                             role_loss="contrast" if loss_type == 'c' else "ranking",
                             tables=tables)
        super().__init__(**parts)
        self.network = parts["classes"].network
        self.role_pairs = role_pairs
        self.thing, self.nothing_reading = thing, nothing
        self.top_index = top_index if thing == 'all' else None
        self.bottom_index = bottom_index if nothing == 'empty' else None
        self.nentities, self.anon_e, self.num_negs = nentities, anon_e, num_negs
        self.t_norm, self.max_measure, self.loss_type = t_norm, max_measure, loss_type
        self.heads_dict, self.tails_dict = heads_dict, tails_dict

    @property
    def fc_0(self):
        return self.network.fc_0

    @property
    def fc_1(self):
        return self.network.fc_1

    def _mem(self, c_emb, e_emb):
        """The membership network: maps a (concept, entity) embedding pair to a fuzzy
        membership degree in ``[0, 1]`` (with a trailing dimension of 1)."""
        return self.network(c_emb, e_emb).unsqueeze(-1)

    def _get_c_fs_batch(self, c_emb, e_emb):
        """The degrees of the concept vectors ``c_emb`` ``(M, D)`` for the entities ``e_emb``
        ``(N, D)``: ``(M, N)``."""
        e_emb = e_emb.unsqueeze(dim=0).repeat(c_emb.size()[0], 1, 1)
        c_emb = c_emb.unsqueeze(dim=1).expand_as(e_emb)
        return self.network(c_emb, e_emb)

    def forward(self, *args, stage='train', **kwargs):
        """The loss of a batch of an axiom pattern (:meth:`ExpressionGeometry.forward`).
        ``stage='test'`` for role assertions gives the degrees of the assertions and of
        their corrupted assertions instead."""
        if stage == 'test' and len(args) > 1 and not isinstance(args[1], str):
            from mowl.owlapi import OWLObjectPropertyAssertionAxiom
            if isinstance(args[0], OWLObjectPropertyAssertionAxiom):
                e_emb = args[2] if len(args) > 2 else kwargs.get("e_emb")
                domain = self._context(e_emb)
                try:
                    return self.objective.role_scores(self, args[1])
                finally:
                    if domain is not None:
                        domain.context = None
        elif stage != 'train':
            raise ValueError("stage must be 'train' or 'test'.")
        return super().forward(*args, **kwargs)
