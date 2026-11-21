"""Default settings for the joint ``common-*`` benchmarks.

These are not the settings of a publication. Where a publication trained a method on the
same data, its recipe is reused (Box2EL and its baselines, TransBox, ELK-Em); every other
method gets plain defaults with a training length suited to how it is trained. They are a
common starting point for a comparison, not tuned results: tune on the validation split
before drawing conclusions about a method.
"""

from dataclasses import replace

from mowl.benchmarks.recipes import Recipe
from mowl.catalog.card import Reference

MOWL_COMMON = Reference(
    "mowl-common",
    "mOWL joint benchmarks: default settings (documentation, section 'Comparing all "
    "methods: the joint benchmarks').",
    "https://mowl.readthedocs.io/en/latest/benchmarks/index.html")

_ONTOLOGIES = ("galen", "go", "anatomy")

#: joint benchmark pattern -> benchmarks whose recipes are reused, in order of preference
_REUSE = {
    "common-{o}-subsumption-prediction": ("box2el-{o}-prediction", "transbox-{o}-prediction",
                                          "elkem-{o}-subsumption"),
    "common-{o}-existential-prediction": ("box2el-{o}-prediction", "transbox-{o}-prediction",
                                          "elkem-{o}-subsumption"),
    "common-{o}-subsumption-inference": ("box2el-{o}-inference", "elkem-{o}-subsumption"),
}

_EL = {"training_regime": "box2el-code-training", "epochs": 5000, "validate_every": 500}
_NOTE = "Untuned default of the joint benchmarks."

#: method -> settings for methods without a reusable recipe
_DEFAULTS = {
    # normal-form EL models: the training procedure of the Box2EL code, shorter
    "emelvar": dict(_EL, model_kwargs={"embed_dim": 200, "load_role_axioms": True}),
    "regd": dict(_EL, model_kwargs={"embed_dim": 200}),
    "regd_elem": dict(_EL, model_kwargs={"embed_dim": 200}),
    "octagon": dict(_EL, model_kwargs={"embed_dim": 200}),
    "prism": dict(_EL, model_kwargs={"embed_dim": 200}),
    "multiboxel": dict(_EL, model_kwargs={"embed_dim": 50}),
    "multielbe": dict(_EL, model_kwargs={"embed_dim": 50}),
    # no loss for corrupted axioms: their own training loop
    "mcmultibox": {"model_kwargs": {}, "epochs": 500},
    "elkem": {"model_kwargs": {"embed_dim": 50, "load_role_axioms": True}, "epochs": 5000},
    "transbox": dict(_EL, model_kwargs={"embed_dim": 200, "load_role_axioms": True}),
    # other trained models: their own loop
    "cate": {"model_kwargs": {"embed_dim": 200}, "epochs": 2000, "validate_every": 100},
    "falcon": {"model_kwargs": {"embed_dim": 50}, "epochs": 100},
    "alcone": {"model_kwargs": {}, "epochs": 500},
    "e2r": {"model_kwargs": {"embed_dim": 100}, "epochs": 500},
    "bcbox": {"model_kwargs": {"embed_dim": 32}, "epochs": 500},
    "dfalc": {"model_kwargs": {"anon_domain": 8}, "epochs": 200},
    "eike": {"model_kwargs": {"embed_dim": 100}, "epochs": 500},
    "boxsel": {"model_kwargs": {"embed_dim": 50}, "epochs": 500},
    # hierarchy embeddings of the subclass order: full-batch Riemannian SGD
    "hyperbolic_cones": {"model_kwargs": {"embed_dim": 10}, "epochs": 1000},
    "shadow_cones": {"model_kwargs": {"embed_dim": 10}, "epochs": 1000},
    "disk_embeddings": {"model_kwargs": {"embed_dim": 10}, "epochs": 1000},
    # needs individuals; fails with that message on the TBox-only data
    "canonical_mixture": {"model_kwargs": {}, "epochs": 500},
    # fixed convex TBox, learned ABox; refuses ontologies whose canonical model is too big
    "faithel": {"model_kwargs": {}, "epochs": 100},
    "grounded": {"model_kwargs": {}, "epochs": 500},
    "graph_kge": {"model_kwargs": {"embed_dim": 200}, "epochs": 100},
    # region-based KGEs of the projection (PyKEEN loop) and the two-view box model
    "boxe": {"model_kwargs": {"embed_dim": 200}, "epochs": 100},
    "expressive": {"model_kwargs": {"embed_dim": 200}, "epochs": 100},
    "reshuffle": {"model_kwargs": {}, "epochs": 50},
    "concept2box": {"model_kwargs": {"embed_dim": 50}, "epochs": 500},
    # query embeddings: one epoch is 10 steps on freshly sampled queries
    "query2box": {"model_kwargs": {"embed_dim": 100}, "epochs": 200},
    "betae": {"model_kwargs": {"embed_dim": 100}, "epochs": 200},
    "cone_qe": {"model_kwargs": {"embed_dim": 100}, "epochs": 200},
    "acone": {"model_kwargs": {"embed_dim": 100}, "epochs": 200},
    # Word2Vec on walks or on a corpus, scored with the pair classifier head
    "dl2vec": {"model_kwargs": {"embed_dim": 100}, "epochs": 10},
    "owl2vec_star": {"model_kwargs": {"embed_dim": 100}, "epochs": 10},
    "taxonomy_walks": {"model_kwargs": {"embed_dim": 100}, "epochs": 10},
    "onto2vec": {"model_kwargs": {"embed_dim": 100}, "epochs": 10},
    "opa2vec": {"model_kwargs": {"embed_dim": 100}, "epochs": 10},
    # language models: one pass over the training triplets
    "hit": {"model_kwargs": {}, "epochs": 1},
    "ont": {"model_kwargs": {}, "epochs": 1},
    # order and box embeddings of the subclass taxonomy (gci0 only): full-batch generic loop
    "order_embeddings": {"model_kwargs": {"embed_dim": 50}, "epochs": 1000},
    "box_lattice": {"model_kwargs": {"embed_dim": 50}, "epochs": 1000},
    "gumbel_box": {"model_kwargs": {"embed_dim": 50}, "epochs": 1000},
}

#: taxonomy methods: no score for C ⊑ ∃r.D, so no existential or PPI recipe
_TAXONOMY_ONLY = ("order_embeddings", "box_lattice", "gumbel_box")


def _build():
    # Reads the recipe modules of the publications directly (not through the registry,
    # which is being built when this module is imported).
    from mowl.benchmarks.recipe_sets import box2el_transbox, singh2026
    published = tuple(box2el_transbox.RECIPES) + tuple(singh2026.RECIPES)
    out = []
    for onto in _ONTOLOGIES:
        for pattern, sources in _REUSE.items():
            benchmark = pattern.format(o=onto)
            done = set()
            for source in sources:
                for recipe in published:
                    if recipe.benchmark != source.format(o=onto) or recipe.method in done \
                            or recipe.name:
                        continue
                    done.add(recipe.method)
                    out.append(replace(
                        recipe, benchmark=benchmark, source=MOWL_COMMON,
                        locator=f"recipe of {recipe.source.key} for {recipe.benchmark}",
                        notes=f"The settings {recipe.source.key} used on the same data "
                              f"({recipe.benchmark}). {recipe.notes}"))
            for method, settings in _DEFAULTS.items():
                if method in done:
                    continue
                if "existential" in benchmark and method in (
                        "bcbox", "taxonomy_walks", "hit", "e2r", "eike") + _TAXONOMY_ONLY:
                    continue
                out.append(Recipe(benchmark=benchmark, method=method, source=MOWL_COMMON,
                                  locator="mowl.benchmarks.recipe_sets.common_defaults",
                                  notes=_NOTE, **settings))
    ppi_skip = {"bcbox", "taxonomy_walks", "hit", "e2r", *_TAXONOMY_ONLY}
    for organism in ("yeast", "human"):
        benchmark = f"common-{organism}-ppi"
        done = set()
        for recipe in published:
            if recipe.benchmark == f"box2el-{organism}-ppi" and not recipe.name \
                    and recipe.method not in done:
                done.add(recipe.method)
                out.append(replace(
                    recipe, benchmark=benchmark, source=MOWL_COMMON,
                    locator=f"recipe of {recipe.source.key} for {recipe.benchmark}",
                    notes=f"The settings {recipe.source.key} used on the same data "
                          f"({recipe.benchmark}). {recipe.notes}"))
        el_baselines = {m: dict(_EL, model_kwargs={"embed_dim": 200})
                        for m in ("elem", "emelpp", "elbe", "box2el")}
        el_baselines["boxel"] = dict(_EL, model_kwargs={"embed_dim": 25})
        for method, settings in {**el_baselines, **_DEFAULTS}.items():
            if method in done or method in ppi_skip:
                continue
            settings = dict(settings)
            if settings.get("training_regime") == "box2el-code-training":
                # The Box2EL procedure selects on validation axioms C ⊑ D, which the
                # interaction data do not have: plain training, last epoch.
                settings = {"model_kwargs": settings["model_kwargs"], "epochs": 5000}
            out.append(Recipe(benchmark=benchmark, method=method, source=MOWL_COMMON,
                              locator="mowl.benchmarks.recipe_sets.common_defaults",
                              notes=_NOTE, **settings))
    return tuple(out)


RECIPES = _build()
