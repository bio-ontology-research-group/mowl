"""Small instances of every ELModule subclass and a batch for each axiom form, used by the
tests and the survey of per-axiom loss shapes."""
import importlib
import pkgutil

import torch as th

N, R, I, D, B = 8, 3, 4, 6, 5  # noqa: E741

FORMS = ("gci0", "gci0_bot", "gci1", "gci1_bot", "gci2", "gci3", "gci3_bot",
         "class_assertion", "object_property_assertion", "role_inclusion", "role_chain")

# The bottom forms are padded with a column that the ELBE-style losses ignore and that the
# BoxEL/TransBox losses read as the (ignored) bottom class.
_HI = {
    "gci0": [N, N], "gci0_bot": [N, N], "gci1": [N, N, N], "gci1_bot": [N, N, N],
    "gci2": [N, R, N], "gci3": [R, N, N], "gci3_bot": [R, N, N],
    "class_assertion": [I, N], "object_property_assertion": [I, R, I],
    "role_inclusion": [R, R], "role_chain": [R, R, R],
}


def batch(form, rows=B, seed=7):
    gen = th.Generator().manual_seed(seed)
    return th.stack([th.randint(0, h, (rows,), generator=gen) for h in _HI[form]], dim=1)


def _canonical_mixture():
    from mowl.geometric.membership import MembershipStructure
    from mowl.nn.el.canonical_mixture.module import CanonicalMixtureModule
    n_cls, n_ind = N, I
    n_el = n_cls + n_ind
    rows = list(range(n_cls)) + [n_cls + i for i in range(n_ind)] * 2
    cols = list(range(n_cls)) + [i for i in range(n_ind)] + [n_cls - 1] * n_ind
    m = th.sparse_coo_tensor(th.tensor([rows, cols]), th.ones(len(rows), dtype=th.bool),
                             size=(n_el, n_cls), dtype=th.bool).coalesce()
    names = [f"c{c}" for c in range(n_cls)] + [f"i{i}" for i in range(n_ind)]
    ms = MembershipStructure(
        element_names=names, element_index={n: k for k, n in enumerate(names)}, M=m, roles={},
        individual_element=th.arange(n_cls, n_el),
        class_index_dict={f"c{c}": c for c in range(n_cls)},
        individual_index_dict={f"i{i}": i for i in range(n_ind)}, role_index_dict={})
    return CanonicalMixtureModule(ms, rank=4)


def _specials():
    from mowl.nn.el.mcmultibox.module import VolumeMultiBoxELModule
    return {
        "CanonicalMixtureModule": _canonical_mixture,
        "VolumeMultiBoxELModule": lambda: VolumeMultiBoxELModule(N, R, I, embed_dim=3,
                                                                 num_boxes=2, num_points=64),
    }


def _kwargs(cls):
    name = cls.__name__
    if name in ("TaxonomyModule", "OrderEmbeddingsModule", "UnitCubeBoxModule",
                "BoxLatticeModule", "GumbelBoxModule"):
        return dict(nb_ont_classes=N, embed_dim=D) if name != "TaxonomyModule" else {
            "nb_ont_classes": N}
    if name in ("OrderModule", "HyperbolicConesModule", "ShadowConesModule",
                "DiskEmbeddingsModule"):
        return dict(nb_classes=N, embed_dim=D)
    if name in ("EIKEModule", "Concept2BoxModule"):
        return dict(nb_classes=N, nb_rels=R, nb_inds=I, embed_dim=D)
    return dict(nb_ont_classes=N, nb_rels=R, nb_inds=I, embed_dim=D)


def all_el_module_classes():
    """Every concrete :class:`~mowl.nn.ELModule` subclass defined under ``mowl.nn``."""
    import mowl.nn
    from mowl.nn import ELModule
    for pkg in (mowl.nn.el, mowl.nn.order):
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            importlib.import_module(m.name)

    def subs(c):
        for s in c.__subclasses__():
            yield s
            yield from subs(s)
    out = {}
    for c in subs(ELModule):
        if c.__module__.startswith("mowl.nn.") and not c.__name__.startswith("_"):
            out[c.__name__] = c
    return [out[k] for k in sorted(out)]


def make(cls):
    th.manual_seed(0)
    special = _specials().get(cls.__name__)
    return special() if special else cls(**_kwargs(cls))
