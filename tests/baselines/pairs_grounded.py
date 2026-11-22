"""Comparisons of ``grounded`` (GroundedEL): the part-built geometry, constraints and
regimes against the direct implementation (``mowl.baselines.models.grounded``, whose module
and hinge losses are its own: ``mowl.baselines.nn.el.grounded``).

Two data sets: the ``roles.owl`` fixture, and a small table of axioms that has every form
(gci0 .. gci3, the bottom forms, role inclusion and chain, class and role assertions, top
as a filler and bottom), so that every constraint and every crisp check is compared. The
variants cover the code paths the options select: the two relation models, the ablations
(``dead_from_bot_only=False``, ``agg="lse"``, ``region_lhs_grad=False``) and the Lagrangian
dual weights with corrupted negatives. ``grounded/queries`` also compares the scores of
logical queries (``score_query``, compiled to unions of boxes and by beam search).
"""
import torch as th

from tests.baselines.harness import Pair


def _roles_dataset():
    from mowl.datasets import PathDataset
    from tests.geometric.helpers import fixture_path
    return PathDataset(fixture_path("roles.owl"))


def _all_forms_tables():
    """Every axiom form over 7 classes (with top and bottom), 3 roles, 3 individuals."""
    from mowl.owlapi.defaults import BOT, TOP
    names = ["A", "B", "C", "D", "E", "F", TOP, BOT]
    c = {n: i for i, n in enumerate(names)}
    r = {"r": 0, "s": 1, "t": 2}
    i = {"a": 0, "b": 1, "c": 2}

    def t(rows):
        return th.tensor(rows, dtype=th.long)
    return dict(
        class_index_dict=c, object_property_index_dict=r, individual_index_dict=i,
        gci0=t([[c["A"], c["B"]], [c["B"], c["C"]]]),
        gci1=t([[c["A"], c["D"], c["E"]]]),
        gci2=t([[c["A"], r["r"], c["B"]], [c["C"], r["s"], c[TOP]]]),
        gci3=t([[r["r"], c["B"], c["D"]], [r["s"], c[TOP], c["E"]]]),
        gci0_bot=t([[c["F"]]]),
        gci1_bot=t([[c["B"], c["F"]]]),
        gci3_bot=t([[r["t"], c["F"]]]),
        role_inclusion=t([[r["r"], r["s"]]]),
        role_chain=t([[r["r"], r["s"], r["t"]]]),
        class_assertion=t([[i["a"], c["A"]]]),
        object_property_assertion=t([[i["a"], r["r"], i["b"]], [i["b"], r["s"], i["c"]]]))


_GROUNDED = dict(embed_dim=3, n_components=2, n_role_components=2, epochs=3, batch_size=16,
                 learning_rate=0.01, n_region_samples=2, n_adv=1, n_random_elements=2,
                 n_inside_elements=4, n_filler_samples=4, seed=0, log_every=0)


def _build(cls, data, options):
    if isinstance(data, dict):
        return cls.from_tables(**data, **{**_GROUNDED, **options})
    return cls(data, **{**_GROUNDED, **options})


def _grounded_new(**options):
    def build(data):
        from mowl.models import GroundedEL
        return _build(GroundedEL, data, options)
    return build


def _grounded_old(**options):
    def build(data):
        from mowl.baselines.models.grounded.model import GroundedEL
        return _build(GroundedEL, data, options)
    return build


def _grounded_copy(old, new):
    # the part-built module loads a state_dict with the table names of the direct module
    new.module.load_state_dict(old.module.state_dict())


def _grounded_losses(model):
    generator = th.Generator().manual_seed(1234)
    out = model.compute_batch_losses(generator=generator)
    return {k: v.detach() for k, v in out.items() if isinstance(v, th.Tensor)}


def _grounded_scores(model):
    """The membership of every element in every class, the crisp report (violation rates)."""
    m = model.module
    positions = m.positions().detach()
    n = positions.shape[0]
    member = th.stack([m.membership(positions, th.full((n,), c, dtype=th.long)).detach()
                       for c in range(m.nb_classes)])
    report = model.crisp_report(n_eval=64)
    return {"membership": member,
            "violations": th.tensor([float(report["elem_viol"]), float(report["region_viol"])])}


_QUERIES = {
    "1p": (("e", ("r",)), [(0, (0,)), (1, (1,))]),
    "2p": (("e", ("r", "r")), [(0, (0, 1))]),
    "2i": ((("e", ("r",)), ("e", ("r",))), [((0, (0,)), (1, (1,)))]),
    "2u": ((("e", ("r",)), ("e", ("r",)), ("u",)), [((0, (0,)), (1, (1,)))]),
    "ip": (((("e", ("r",)), ("e", ("r",))), ("r",)), [(((0, (0,)), (1, (0,))), (1,))]),
    "pi": ((("e", ("r", "r")), ("e", ("r",))), [((0, (0, 1)), (1, (1,)))]),
}


def _grounded_query_scores(model):
    """The scores of :func:`_grounded_scores` and those of logical queries of several
    structures, compiled to unions of boxes and by the beam search."""
    out = _grounded_scores(model)
    for name, (structure, queries) in _QUERIES.items():
        for method in ("box", "beam"):
            out[f"query/{name}/{method}"] = model.score_query(structure, queries,
                                                              method=method).detach()
    return out


def _grounded_train(model, epochs):
    model.train(epochs=epochs)


def _grounded_parameters(model):
    """The tables in the naming of the direct module (the part-built module names its
    tables by part: ``boxes.centers`` is ``class_centers``)."""
    from mowl.nn.el.grounded.module import _LEGACY_KEYS
    to_old = {new: old for old, new in _LEGACY_KEYS.items()}
    out = {}
    for key, value in model.module.state_dict().items():
        for new, old in to_old.items():
            if key == new or key.startswith(new + "."):
                key = old + key[len(new):]
                break
        out[key] = value.detach()
    return out


def _grounded_pair(key, dataset=_roles_dataset, scores=_grounded_scores, **options):
    return Pair(key=key, new=_grounded_new(**options), old=_grounded_old(**options),
                copy=_grounded_copy, scores=scores, losses=_grounded_losses,
                train=_grounded_train, parameters=_grounded_parameters, dataset=dataset,
                epochs=3)


PAIRS = [
    _grounded_pair("grounded"),
    _grounded_pair("grounded/translation", role_mode="translation", neg_weight=0.5),
    _grounded_pair("grounded/all-forms", dataset=_all_forms_tables),
    _grounded_pair("grounded/all-forms-translation", dataset=_all_forms_tables,
                   role_mode="translation"),
    _grounded_pair("grounded/ablations", dataset=_all_forms_tables, dead_from_bot_only=False,
                   agg="lse", region_lhs_grad=False),
    _grounded_pair("grounded/lagrangian", dataset=_all_forms_tables,
                   constraint_mode="lagrangian", neg_weight=0.5, role_mode="translation",
                   agg="lse"),
    # the query algebra over unions of boxes (library: a general part; baseline: its copy)
    _grounded_pair("grounded/queries", dataset=_all_forms_tables, role_mode="translation",
                   scores=_grounded_query_scores),
]
EXEMPT = {}
