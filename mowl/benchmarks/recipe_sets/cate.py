"""Recipes of CatE (Zhapa-Camacho and Hoehndorf, arXiv:2305.07163v3, Tables 2-4), from
the paper's Appendix A, Table 6 and the reference code (bio-ontology-research-group/
catE, use_cases/run_*.py)."""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import ZHAPA2023

_LOC = ("Table 6 of the paper (Appendix A) for embedding size, learning rate, margin, "
        "batch size and negatives; the run scripts of the reference code for the "
        "number of epochs (10,000; 50,000 for ORE1) and the validation interval (100); "
        "src/cat.py for the schedule, the weight decay and model selection")
_NOTE = ("The loss is Eq. 7 of the paper as the reference code computes it (the "
         "negatives of one morphism are averaged before the hinge, and corruptions "
         "that are morphisms are kept); the code also offers a log-sigmoid loss "
         "(loss_type='bpr') that is not catalogued. The norm p is not in Table 6: the "
         "paper's Eq. 6 prints the L2 norm, the ORE1 run script defaults to p=1, and "
         "the released PPI run script passes its arguments in the wrong order, so the "
         "norms actually used are unknown; the recipes use p=2. Early stopping: the "
         "code validates every 100 epochs and stops after 10 non-improving "
         "validations.")

#: benchmark -> (dim, lr, margin, batch_size, num_negs, epochs)
_CATE = {
    "cate-ore1-membership": (200, 1e-4, 1, 32768, 4, 50000),
    "cate-go-completion": (200, 1e-5, 1, 32768, 4, 10000),
    "cate-foodon-completion": (200, 1e-4, 1, 8192, 2, 10000),
    "cate-yeast-ppi": (256, 1e-4, 0.1, 2048, 4, 10000),
}

#: The released training graphs of ORE1, GO and FoodOn were produced by the reference
#: projector (no negation completion) and then filtered (remove_obvious_morphisms.py
#: drops the trivial morphisms of complex elements).
_RELEASED_GRAPH = {"complete_negation": False, "obvious_morphisms": False}


def _build():
    out = []
    for benchmark, (dim, lr, margin, bs, negs, epochs) in _CATE.items():
        kwargs = {}
        model_kwargs = {"embed_dim": dim, "learning_rate": lr, "margin": margin,
                        "batch_size": bs, "num_negs": negs}
        notes = _NOTE
        if benchmark != "cate-yeast-ppi":
            model_kwargs.update(_RELEASED_GRAPH)
        if benchmark == "cate-yeast-ppi":
            # The released PPI training graph is ontology_extended.cat.s1.edgelist:
            # the lattice extended with the protein existentials and one round of the
            # S1 saturation rules; the PPI term of the training loop is not described
            # in the paper. The file name carries no "_filtered", so the trivial
            # morphisms are kept (obvious_morphisms stays True); as everywhere, the
            # reference projector does not complete negations.
            model_kwargs["saturation_steps"] = 1
            model_kwargs["complete_negation"] = False
            kwargs["interaction_loss"] = True
            notes = (_NOTE + " The lattice is the S1-saturated extended lattice of the "
                     "code's graph file (ontology_extended.cat.s1.edgelist) and the "
                     "training loop adds the code's interaction loss (protein-headed "
                     "morphisms against corrupted existential proteins).")
        out.append(Recipe(
            benchmark=benchmark, method="cate", source=ZHAPA2023,
            model_kwargs=model_kwargs, training_regime="cate-code-training",
            regime_kwargs=kwargs, epochs=epochs, validate_every=100,
            locator=_LOC, notes=notes))
    return tuple(out)


RECIPES = _build()
