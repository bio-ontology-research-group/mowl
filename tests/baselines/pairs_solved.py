"""Comparisons of the solved methods: ``boxlite``, ``faithel`` (``grounded``:
``pairs_grounded.py``).

BoxLitE is solved by a conic solver; the optional package ``cvxpy`` is not always installed.
Its comparison therefore sets the embedding of the direct implementation by hand (the same
random box interpretation in both), compares the losses, the scores and the verdicts on the
TBox, and runs the solver run whenever cvxpy can be imported (a skip with a reason
otherwise; with the environment variable ``MOWL_REQUIRE_CVXPY=1``, as set by the CI job that
installs cvxpy, a missing solver is a failure, so the job cannot pass without the
comparison). The solved boxes of both sides are compared as well as the scores.
"""
import os

import numpy as np
import pytest
import torch as th

from tests.baselines.harness import Pair

# ---------------------------------------------------------------------------------------
# BoxLitE
# ---------------------------------------------------------------------------------------
_BOXLITE_KB = dict(
    inclusions=[("A", "B"), ("∃s-", "P"), ("B", "∃r")],
    negative_inclusions=[("P", "∃t"), ("B", "D")],
    role_inclusions=[("r", "s-"), ("t", "t-"), ("u", "r")],
    concept_assertions=[("∃u", "a"), ("A", "b"), ("D", "e")],
    role_assertions=[("u", "a", "b"), ("t", "c", "d"), ("r", "b", "e")])


def _kb():
    from mowl.datasets.dllite import DLLiteKB
    return DLLiteKB.from_lists(**_BOXLITE_KB)


_DIM = None   # the dimension of Corollary 2 for the KB (set below)


def _dim(side="new"):
    """The dimension of Corollary 2, each side by its own code."""
    if side == "old":
        from mowl.baselines.models.boxlite.model import kb_faithful_dimension
    else:
        from mowl.representation.library.solved_boxlite import kb_faithful_dimension
    return kb_faithful_dimension(_kb())


def _boxlite_old(variant):
    def build(_dataset):
        from mowl.baselines.models.boxlite.embedding import BoxLitEEmbedding
        from mowl.baselines.models.boxlite.model import BoxLitE
        kb = _kb()
        model = BoxLitE.from_kb(kb, embed_dim=_dim("old"), variant=variant)
        rng = np.random.default_rng(1)
        d = _dim("old")
        ni, nc, nr = len(kb.individuals), len(kb.concepts), len(kb.roles)

        def boxes(n):
            lower = rng.uniform(-0.8, 0.0, (n, d))
            return lower, lower + rng.uniform(0.2, 1.0, (n, d))

        concept, head, tail, bump = boxes(nc), boxes(nr), boxes(nr), boxes(nr)
        model.embedding = BoxLitEEmbedding(
            pos=rng.uniform(-0.5, 0.5, (ni, d)), bump=rng.uniform(-0.5, 0.5, (ni, d)),
            concept_lower=concept[0], concept_upper=concept[1],
            head_lower=head[0], head_upper=head[1], tail_lower=tail[0], tail_upper=tail[1],
            bump_lower=bump[0], bump_upper=bump[1], s_omega=1.0,
            margin=0.01 if variant == "paper" else 0.0)
        return model
    return build


def _boxlite_new(variant):
    def build(_dataset):
        from mowl.models import BoxLitE
        return BoxLitE.from_kb(_kb(), embed_dim=_dim(), variant=variant)
    return build


def _boxlite_copy(old, new):
    new.module.load_values({k: getattr(old.embedding, k) for k in old.embedding._ARRAYS})
    new.embedding = new.module.snapshot()


def _is_new(model):
    """The combination of parts (its module is a Geometry), not the direct implementation."""
    from mowl.representation.geometry import Geometry
    return isinstance(getattr(model, "module", None), Geometry)


def _assertion_rows(model):
    kb = model.kb
    ni, nb, nr = len(kb.individuals), kb.n_basic, len(kb.roles)
    concept = th.tensor([[a, b] for a in range(ni) for b in range(nb)])
    role = th.tensor([[a, r, b] for r in range(nr) for a in range(ni) for b in range(ni)])
    return concept, role


def _boxlite_losses(model):
    """The member loss of every (individual, basic concept) and role triple; the number of
    violated TBox axioms of each kind."""
    kb = model.kb
    concept, role = _assertion_rows(model)
    out = {}
    if _is_new(model):
        g = model.module
        out["class_assertion"] = g.cost("class_assertion", concept)
        out["object_property_assertion"] = g.cost("object_property_assertion", role)
        out["class_assertion/neg"] = g.cost("class_assertion", concept, neg=True)
        nr = len(kb.roles)
        ci = th.as_tensor(kb.concept_inclusions)
        ni = th.as_tensor(kb.negative_inclusions)
        ri = th.as_tensor(kb.role_inclusions)
        signed = th.stack([ri[:, 0] + nr * ri[:, 1], ri[:, 2] + nr * ri[:, 3]], dim=1)
        tol = 1e-6
        out["violated"] = th.tensor([
            float((g.cost("gci0", ci) > tol).sum()),
            float((g.negative_inclusion_cost(ni) > tol).sum()),
            float((g.cost("role_inclusion", signed) > tol).sum())])
    else:
        e = model.embedding
        out["class_assertion"] = th.tensor(
            [[e.concept_loss(int(b), [int(a)])[0]] for a, b in concept.tolist()])
        out["object_property_assertion"] = th.tensor(
            [[e.role_loss(int(r), [int(a)], [int(b)])[0]] for a, r, b in role.tolist()])
        from mowl.baselines.models.boxlite.solved_check import check_model
        out["class_assertion/neg"] = th.tensor(
            [[_negative_loss(e, int(b), int(a))] for a, b in concept.tolist()])
        check = check_model(e, kb, tol=1e-6, max_reports=10 ** 6)
        out["violated"] = th.tensor([
            float(len([x for x in check.violations.get(k, []) if x != "..."]))
            for k in ("concept_inclusion", "negative_inclusion", "role_inclusion")])
    return out


def _negative_loss(embedding, basic, individual):
    """Signed distance of a position to the complement box (direct formula, NumPy)."""
    from mowl.baselines.models.boxlite.embedding import box_distance
    lower, upper = embedding.basic_bounds()
    cl, cu = embedding.complement_bounds(lower[basic], upper[basic])
    return float(box_distance(cl, cu, embedding.pos[individual], embedding.margin))


def _boxlite_scores(model):
    concept, role = _assertion_rows(model)
    if _is_new(model):
        g = model.module
        return {"concept": -g.cost("class_assertion", concept).detach(),
                "role": -g.cost("object_property_assertion", role).detach()}
    e = model.embedding
    return {"concept": th.tensor([[-e.concept_loss(int(b), [int(a)])[0]]
                                  for a, b in concept.tolist()]),
            "role": th.tensor([[-e.role_loss(int(r), [int(a)], [int(b)])[0]]
                               for a, r, b in role.tolist()])}


def _boxlite_train(model, epochs):
    """The solver run: needs cvxpy."""
    import importlib.util
    if importlib.util.find_spec("cvxpy") is None:
        if os.environ.get("MOWL_REQUIRE_CVXPY") == "1":
            pytest.fail("MOWL_REQUIRE_CVXPY=1 but cvxpy is not installed")
        pytest.skip("the optional package cvxpy is not installed, so the BoxLitE programme "
                    "cannot be solved; install it to compare the solved embeddings")
    model.train(solver="CLARABEL" if "CLARABEL" in _installed() else None)


def _installed():
    import cvxpy
    return set(cvxpy.installed_solvers())


def _boxlite_parameters(model):
    """The boxes and positions of the embedding (solved ones after training)."""
    e = model.module.snapshot() if _is_new(model) else model.embedding
    return {name: th.as_tensor(np.asarray(getattr(e, name)), dtype=th.float64)
            for name in e._ARRAYS}


def _boxlite_pair(variant):
    key = "boxlite" if variant == "paper" else "boxlite/reference"
    return Pair(key=key, new=_boxlite_new(variant), old=_boxlite_old(variant),
                copy=_boxlite_copy, scores=_boxlite_scores, losses=_boxlite_losses,
                train=_boxlite_train, parameters=_boxlite_parameters, dataset=lambda: None,
                epochs=1)


# ---------------------------------------------------------------------------------------
# FaithEL
# ---------------------------------------------------------------------------------------
def _faithel_dataset():
    import os
    import pathlib
    import tempfile
    from tests.models.test_canonical_mixture import _write_family_abox_dataset
    base = os.environ.get("TMPDIR") or None
    return _write_family_abox_dataset(pathlib.Path(tempfile.mkdtemp(prefix="mowl_faithel_",
                                                                      dir=base)))


def _faithel_new(**options):
    # the authors' loss is an explicit option: the default of the library is the bounded hinge
    options.setdefault("negative_term", "unbounded")

    def build(dataset):
        from mowl.models import FaithEL
        return FaithEL(dataset, batch_size=8, learning_rate=0.01, seed=0, **options)
    return build


def _faithel_old(**options):
    def build(dataset):
        from mowl.baselines.models.faithel.model import FaithEL
        return FaithEL(dataset, batch_size=8, learning_rate=0.01, seed=0, **options)
    return build


def _faithel_tables(model):
    """The learned tables by a name common to both sides."""
    if _is_new(model):
        m = model.module
        return {"individuals": m.individual_model.centres.weight, "concepts": m.concepts.weight,
                "role_head": m.role_head.weight, "role_tail": m.role_tail.weight}
    m = model.module
    return {"individuals": m.individuals.weight, "concepts": m.concepts.weight,
            "role_head": m.role_head.weight, "role_tail": m.role_tail.weight}


def _faithel_copy(old, new):
    with th.no_grad():
        for name, table in _faithel_tables(new).items():
            table.copy_(_faithel_tables(old)[name])
        a, b = old.module, new.module
        assert th.equal(a.concept_centroid, b.concept_centroid)
        assert th.equal(a.role_centroid, b.role_centroid)


def _faithel_rows(model):
    data = {n: model.training_datasets[n][:] for n in ("class_assertion",
                                                       "object_property_assertion")
            if n in model.training_datasets and len(model.training_datasets[n])}
    return data


def _faithel_losses(model):
    out = {}
    for name, rows in _faithel_rows(model).items():
        if _is_new(model):
            out[name] = model.module.cost(name, rows).mean()
        else:
            out[name] = (model._concept_loss if name == "class_assertion"
                         else model._role_loss)(rows)
    return out


def _faithel_scores(model):
    n, m = len(model.class_index_dict), len(model.object_property_index_dict)
    k = len(model.individual_index_dict)
    pairs = th.tensor([[a, b] for a in range(n) for b in range(n)])
    triples = th.tensor([[a, b, c] for a in range(min(n, 6)) for b in range(min(n, 6))
                         for c in range(min(n, 6))])
    gci2 = th.tensor([[a, r, b] for a in range(n) for r in range(m) for b in range(n)])
    gci3 = th.tensor([[r, a, b] for r in range(m) for a in range(n) for b in range(n)])
    cls = th.tensor([[a, c] for a in range(k) for c in range(n)])
    role = th.tensor([[a, r, b] for a in range(k) for r in range(m) for b in range(k)])
    return {"gci0": model.axiom_costs("gci0", pairs), "gci1": model.axiom_costs("gci1", triples),
            "gci2": model.axiom_costs("gci2", gci2), "gci3": model.axiom_costs("gci3", gci3),
            "class_assertion": model.axiom_costs("class_assertion", cls),
            "object_property_assertion": model.axiom_costs("object_property_assertion", role)}


def _faithel_pair(key, **options):
    return Pair(key=key, new=_faithel_new(**options), old=_faithel_old(**options),
                copy=_faithel_copy, scores=_faithel_scores, losses=_faithel_losses,
                parameters=lambda model: {k: v.detach() for k, v in
                                          _faithel_tables(model).items()},
                dataset=_faithel_dataset, epochs=3)


PAIRS = [_boxlite_pair("paper"), _boxlite_pair("reference"),
         _faithel_pair("faithel"),
         _faithel_pair("faithel/anchor-only", centroid_score=False, neg_sampling=False)]
EXEMPT = {}
