"""Recipes of RESHUFFLE on the inductive benchmarks (``kg-inductive-*``): the best
configurations of Pavlovic, Sallinger and Schockaert (KR 2025; arXiv:2406.09529), appendix
Table "hyperparas" (RESHUFFLE's best-performing hyper-parameters), which are the files
``Best_Configurations/RESHUFFLE/<FB|WN|NELL>_v<n>.json`` of the authors' code
(https://github.com/AleksVap/RESHUFFLE; the two agree). The ablations reuse the
hyper-parameters of the main model ("we reuse the same hyperparameters for each of
RESHUFFLE's ablations"; the files ``RESHUFFLE_no_loop`` and ``RESHUFFLE_square`` do too).

One recipe per benchmark and variant; the names are ``reshuffle``, ``no-loop`` (RESHUFFLE_nL:
``aggregation="max_self_noLoop"``) and ``square`` (RESHUFFLE^2: ``row_function="square"``), so
that ``--recipe pavlovic2025:reshuffle`` selects the main model.
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import PAVLOVIC2025

# (num_layers, l, k, margin, learning rate); appendix Table "hyperparas"
_HYPER = {
    ("fb15k237", "v1"): (4, 25, 80, 2.0, 0.005), ("fb15k237", "v2"): (3, 30, 60, 1.0, 0.005),
    ("fb15k237", "v3"): (5, 25, 40, 0.5, 0.005), ("fb15k237", "v4"): (3, 30, 80, 1.0, 0.01),
    ("wn18rr", "v1"): (3, 20, 40, 1.0, 0.01), ("wn18rr", "v2"): (3, 20, 60, 0.5, 0.01),
    ("wn18rr", "v3"): (3, 20, 40, 1.0, 0.01), ("wn18rr", "v4"): (3, 30, 80, 1.0, 0.01),
    ("nell995", "v1"): (3, 20, 80, 2.0, 0.005), ("nell995", "v2"): (4, 30, 60, 2.0, 0.01),
    ("nell995", "v3"): (4, 25, 40, 0.5, 0.01), ("nell995", "v4"): (4, 30, 60, 1.0, 0.01),
}
_FILE = {"fb15k237": "FB", "wn18rr": "WN", "nell995": "NELL"}

_NOTES = (
    "Trained with PyKEEN's sLCWA loop on the projected training graph (h ⊑ ∃r.t gives the "
    "triple back), Adam, margin ranking loss, batch size 1024, 100 uniformly corrupted "
    "negatives, up to 1000 epochs with early stopping on the validation triples of the "
    "training graph (Hits@10 every 10 epochs, patience 10 evaluations, relative delta 0.01; "
    "the paper says the same in epochs: no improvement of 1% within 100 epochs), best weights "
    "restored. Differences of mOWL's implementation: the random entity features of training "
    "are drawn once per model, not at every forward pass (also at test time: one draw per "
    "evaluation seed); the test negatives are drawn with a seed. inverse_triples=True is the "
    "authors' create_inverse_triples=True (inverse triples as extra positives, heads "
    "predicted through the inverse matrix). Seeds: the paper reports the mean of seeds 1, 2, "
    "3.")

_DIRECTORY = {"reshuffle": "RESHUFFLE", "no-loop": "RESHUFFLE_no_loop",
              "square": "RESHUFFLE_square"}
_VARIANTS = (("reshuffle", {}, "RESHUFFLE"),
             ("no-loop", {"aggregation": "max_self_noLoop"}, "RESHUFFLE_nL"),
             ("square", {"row_function": "square"}, "RESHUFFLE^2"))

RECIPES = tuple(
    Recipe(
        benchmark=f"kg-inductive-{data}-{version}", method="reshuffle", source=PAVLOVIC2025,
        name=name,
        model_kwargs={"l": l, "k": k, "num_layers": layers, "learning_rate": lr,
                      "batch_size": 1024, "num_negs_per_pos": 100, "inverse_triples": True,
                      "early_stopping": {"frequency": 10, "patience": 10,
                                         "relative_delta": 0.01},
                      "kge_kwargs": {"loss_kwargs": {"margin": margin, "reduction": "mean"},
                                     **extra}},
        epochs=1000,
        locator=(f"Appendix, Table hyperparas ({data} {version}); authors' code, "
                 f"Best_Configurations/{_DIRECTORY[name]}/{_FILE[data]}_{version}.json "
                 f"({label})"),
        notes=_NOTES)
    for (data, version), (layers, l, k, margin, lr) in _HYPER.items()  # noqa: E741
    for name, extra, label in _VARIANTS)
