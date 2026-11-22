"""Comparisons of the logic group: DF-ALC, FALCON, Al-cone and E2R, each built from parts
(:mod:`mowl.representation.library.logic_fuzzy`, ``logic_subspace``) against its direct,
hand-written implementation (:mod:`mowl.baselines`).

The ontologies are tiny. The ALC ontology has every shape of axiom the three fuzzy methods
read (inclusions with ``⊓ ⊔ ¬ ∃ ∀``, an equivalence, a disjointness, ``⊤`` and ``⊥``, class
assertions with named and compound classes, role assertions); the E2R ontology the axioms E2R
reads (hierarchy, definitions as meet, join and complement, disjointness, a range, role
inclusion, assertions).
"""
import torch as th

from tests.baselines.harness import Pair

NS = "http://mowl.borg/logic#"


def _ontology(builder):
    from java.util import HashSet
    from mowl.datasets import Dataset
    from mowl.owlapi import OWLAPIAdapter
    ad = OWLAPIAdapter()
    ont = ad.create_ontology(NS + "ontology")
    axioms = HashSet()
    builder(ad, axioms)
    ad.owl_manager.addAxioms(ont, axioms)
    return Dataset(ont)


def alc_dataset():
    def build(ad, axioms):
        c = {n: ad.create_class(NS + n) for n in "ABCDE"}
        thing = ad.create_class("http://www.w3.org/2002/07/owl#Thing")
        nothing = ad.create_class("http://www.w3.org/2002/07/owl#Nothing")
        r, s = ad.create_object_property(NS + "r"), ad.create_object_property(NS + "s")
        some, every = ad.create_object_some_values_from, ad.create_object_all_values_from
        inter, union = ad.create_object_intersection_of, ad.create_object_union_of
        neg, sub = ad.create_complement_of, ad.create_subclass_of
        A, B, C, D, E = (c[n] for n in "ABCDE")
        for axiom in (
                sub(A, B), sub(inter(B, C), D), sub(A, some(r, B)), sub(some(r, C), D),
                sub(every(r, A), B), sub(A, every(s, neg(C))), sub(inter(A, neg(B)), nothing),
                sub(thing, every(r, D)), sub(some(s, some(r, A)), union(B, E)),
                ad.create_equivalent_classes(A, union(B, neg(C))),
                ad.create_disjoint_classes(B, C)):
            axioms.add(axiom)
        x, y, z = (ad.create_individual(NS + n) for n in "xyz")
        for axiom in (
                ad.create_class_assertion(A, x), ad.create_class_assertion(B, y),
                ad.create_class_assertion(C, z), ad.create_class_assertion(some(r, C), z),
                ad.create_class_assertion(inter(A, B), y),
                ad.create_object_property_assertion(r, x, y),
                ad.create_object_property_assertion(r, y, z),
                ad.create_object_property_assertion(s, x, z)):
            axioms.add(axiom)
    return _ontology(build)


def e2r_dataset():
    def build(ad, axioms):
        c = {n: ad.create_class(NS + n) for n in "ABCDEFGH"}
        r, s = ad.create_object_property(NS + "r"), ad.create_object_property(NS + "s")
        A, B, C, D, E, F, G, H = (c[n] for n in "ABCDEFGH")
        sub = ad.create_subclass_of
        for axiom in (
                sub(A, B), sub(B, C), sub(D, C), sub(E, D), sub(F, D),
                ad.create_equivalent_classes(G, ad.create_object_intersection_of(A, D)),
                ad.create_equivalent_classes(H, ad.create_object_union_of(E, F)),
                ad.create_disjoint_classes(A, E),
                sub(B, ad.create_complement_of(F)),
                ad.data_factory.getOWLSubObjectPropertyOfAxiom(r, s),
                ad.data_factory.getOWLObjectPropertyRangeAxiom(r, C)):
            axioms.add(axiom)
        x, y, z, w = (ad.create_individual(NS + n) for n in "xyzw")
        for axiom in (
                ad.create_class_assertion(A, x), ad.create_class_assertion(E, y),
                ad.create_class_assertion(D, z), ad.create_class_assertion(F, w),
                ad.create_object_property_assertion(r, x, y),
                ad.create_object_property_assertion(s, z, w)):
            axioms.add(axiom)
    return _ontology(build)


def _ctx_free(model):
    """The entity context of one evaluation: none for the methods without sampled entities."""
    return None


def _fixed_context(model):
    th.manual_seed(123)
    return model._entity_context(model.sample_anonymous_entities())


def pattern_losses(per_row=True):
    """``model -> {pattern: loss}`` for the rows of every axiom pattern of the training data,
    on a fixed entity context; the mean over the rows where the method averages."""
    def losses(model):
        model._load_datasets()
        context = _fixed_context(model)
        out = {}
        for axiom, data in model._training_datasets.items():
            value = model.module(axiom, data.tensors[0], context)
            out[str(axiom)] = value if per_row else value.mean()
        return out
    return losses


def alc_scores(model):
    """What the rank-based evaluation regimes use: the cost of ``A ⊑ B`` for all pairs of
    classes, of ``A ⊑ ∃r.B`` and ``∃r.A ⊑ B``, and of ``A ⊓ B ⊑ C``."""
    n, k = len(model.class_index_dict), len(model.object_property_index_dict)
    classes = th.arange(n)
    pairs = th.cartesian_prod(classes, classes)
    out = {"gci0": model.axiom_costs("gci0", pairs)}
    triples = th.cartesian_prod(classes[:4], th.arange(k), classes[:4])
    out["gci2"] = model.axiom_costs("gci2", triples)
    out["gci3"] = model.axiom_costs("gci3", th.stack([triples[:, 1], triples[:, 0],
                                                      triples[:, 2]], 1))
    out["gci1"] = model.axiom_costs("gci1", th.cartesian_prod(classes[:4], classes[:4],
                                                              classes[:4]))
    return out


def _copy(old, new, pairs):
    """Copy every table of ``old`` into the table of ``new`` named by ``pairs`` (old name,
    new name), both attribute paths from the module."""
    def get(module, path):
        for part in path.split("."):
            module = getattr(module, part)
        return module
    with th.no_grad():
        for old_name, new_name in pairs:
            get(new.module, new_name).copy_(get(old.module, old_name))


# -- DF-ALC ---------------------------------------------------------------------------------

def _dfalc_copy(old, new):
    _copy(old, new, [("c_embedding.weight", "class_model.centres.weight"),
                     ("r_embedding.weight", "relation_model.vectors.weight")])


def _dfalc_params(model):
    return {"classes": model.module.c_embedding.weight, "roles": model.module.r_embedding.weight}


def _dfalc(loss_type):
    def new(ds):
        from mowl.models import DFALCModel
        return DFALCModel(ds, anon_domain=1, loss_type=loss_type, learning_rate=0.05)

    def old(ds):
        from mowl.baselines.models.dfalc.model import DFALCModel
        return DFALCModel(ds, anon_domain=1, loss_type=loss_type, learning_rate=0.05)
    return new, old


def _dfalc_pair(variant, loss_type):
    new, old = _dfalc(loss_type)
    return Pair(key="dfalc" + variant, new=new, old=old, copy=_dfalc_copy, scores=alc_scores,
                losses=pattern_losses(), parameters=_dfalc_params, dataset=alc_dataset)


# -- FALCON ---------------------------------------------------------------------------------

def _falcon_copy(old, new):
    _copy(old, new, [("c_embedding.weight", "class_model.centres.weight"),
                     ("r_embedding.weight", "relation_model.vectors.weight"),
                     ("e_embedding.weight", "domain.table.weight"),
                     ("fc_0.weight", "network.fc_0.weight"), ("fc_0.bias", "network.fc_0.bias"),
                     ("fc_1.weight", "network.fc_1.weight"),
                     ("fc_1.bias", "network.fc_1.bias")])


def _falcon_params(model):
    m = model.module
    return {"classes": m.c_embedding.weight, "roles": m.r_embedding.weight,
            "entities": m.e_embedding.weight, "fc_0": m.fc_0.weight, "fc_1": m.fc_1.weight}


def _falcon_pair(variant, **options):
    def new(ds):
        from mowl.models import FALCONModel
        return FALCONModel(ds, embed_dim=6, anon_e=2, num_negs=2, learning_rate=0.01,
                           **options)

    def old(ds):
        from mowl.baselines.models.falcon.model import FALCONModel
        return FALCONModel(ds, embed_dim=6, anon_e=2, num_negs=2, learning_rate=0.01,
                           **options)
    return Pair(key="falcon" + variant, new=new, old=old, copy=_falcon_copy,
                scores=alc_scores, losses=pattern_losses(per_row=False),
                parameters=_falcon_params, dataset=alc_dataset)


# -- Al-cone --------------------------------------------------------------------------------

def _alcone_copy(old, new):
    _copy(old, new, [("c_embedding.weight", "class_model.centres.weight"),
                     ("r_embedding.weight", "relation_model.vectors.weight"),
                     ("e_embedding.weight", "individual_model.centres.weight")])


def _alcone_params(model):
    m = model.module
    return {"classes": m.c_embedding.weight, "roles": m.r_embedding.weight,
            "individuals": m.e_embedding.weight}


def _alcone_pair(variant, **options):
    def new(ds):
        from mowl.models import ALConeModel
        return ALConeModel(ds, dim=3, **options)

    def old(ds):
        from mowl.baselines.models.alcone.model import ALConeModel
        return ALConeModel(ds, dim=3, **options)
    return Pair(key="alcone" + variant, new=new, old=old, copy=_alcone_copy,
                scores=alc_scores, losses=pattern_losses(), parameters=_alcone_params,
                dataset=alc_dataset)


# -- E2R ------------------------------------------------------------------------------------

def _e2r_copy(old, new):
    _copy(old, new, [("c_embedding.weight", "class_model.centres.weight"),
                     ("r_embedding.weight", "relation_model.vectors.weight"),
                     ("e_embedding.weight", "individual_model.centres.weight")])


def _e2r_losses(model):
    out = {}
    for key, rows in sorted(model.training_rows.items()):
        kind = model._kind(key)
        out[key] = model.module(kind, rows)
        if kind in ("class_assertion", "role_assertion"):
            out[key + "/neg"] = model.module(kind, rows, neg=True)
    th.manual_seed(5)
    out["objective"] = model.compute_loss(full=True).reshape(1)
    return out


def _e2r_scores(model):
    n_cls, n_ind = len(model.class_index_dict), len(model.individual_index_dict)
    classes = th.arange(n_cls)
    out = {"gci0": model.axiom_costs("gci0", th.cartesian_prod(classes, classes)),
           "class_assertion": model.axiom_costs(
               "class_assertion", th.cartesian_prod(th.arange(n_ind), classes))}
    for iri in sorted(model.class_index_dict):
        out["members " + iri] = th.as_tensor(model.score_membership(iri))
    out["role"] = th.as_tensor(model.score_role_membership(
        sorted(model.object_property_index_dict)[0],
        [(a, b) for a in sorted(model.individual_index_dict)
         for b in sorted(model.individual_index_dict)]))
    return out


def _e2r_params(model):
    m = model.module
    return {"classes": m.c_embedding.weight, "relations": m.r_embedding.weight,
            "entities": m.e_embedding.weight}


def _e2r_pair():
    def new(ds):
        from mowl.models import E2R
        return E2R(ds, embed_dim=5, learning_rate=0.05, sep_gap=1.0)

    def old(ds):
        from mowl.baselines.models.e2r.model import E2R
        return E2R(ds, embed_dim=5, learning_rate=0.05, sep_gap=1.0)
    return Pair(key="e2r", new=new, old=old, copy=_e2r_copy, scores=_e2r_scores,
                losses=_e2r_losses, parameters=_e2r_params, dataset=e2r_dataset)


PAIRS = [
    _dfalc_pair("", "hierarchical"),
    _dfalc_pair("/rule", "rule"),
    _falcon_pair(""),
    _falcon_pair("/minmax", t_norm="minmax"),
    _falcon_pair("/self-pairs", role_pairs="self"),
    _falcon_pair("/learned-nothing", nothing="learned", thing="all"),
    _alcone_pair(""),
    _alcone_pair("/old-constants", constant_logit=1e4),
    _e2r_pair(),
]

EXEMPT = {
    "composed-alc": "ComposedALC is the generic combination of parts for ALC axioms itself (the "
                    "named methods dfalc, falcon and alcone are compared); it has no direct "
                    "implementation to be compared with.",
" Its trained counterpart, alcone, is compared.",
}
