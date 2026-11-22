"""Regime cards: the training of the graph-style hierarchy methods (BCBox, CatE) as parts."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import ZHANG2022, ZHAPA2023, DASGUPTA2020

_FOR = "Models trained by the loop of mowl.training.loop or by the generic EL loop"

REGIMES = (
    RegimeCard(
        key="epoch-batches",
        name="Epochs as passes over a fresh permutation, in minibatches",
        kind=RegimeKind.TRAINING,
        summary=("Every epoch shuffles the rows of every form and takes one optimisation step "
                 "per minibatch of consecutive rows (the last one smaller); an epoch has "
                 "max_f ceil(n_f / batch size) steps."),
        components=("batching",),
        applies_to=_FOR,
        references=(ZHANG2022, ZHAPA2023),
        standalone=False,
        introduced_with=("bcbox", "cate"),
        used_by=("hit", "ont"),
        implementation="mowl.training.graph.EpochBatches",
        notes=("BCBox (Zhang et al. 2022, Section 4.4: minibatches of positive edges) and the "
               "CatE code (a pass over all morphisms in random order). Unlike "
               "per-normal-form-minibatch the batches of an epoch do not repeat rows. The "
               "permutation comes from torch.randperm, or from a shuffle by an object of the "
               "model (the python generator of HiT)."),
    ),
    RegimeCard(
        key="edge-corruption",
        assumption="closed world over stated axioms",
        name="Corruptions of one or two columns, drawn at once",
        kind=RegimeKind.TRAINING,
        summary=("Every row gets k corruptions, each replacing one entity by a uniformly "
                 "drawn one: one fixed column, or one of two columns decided by a coin. All "
                 "draws are made at once from the global torch generator."),
        components=("negative sampling",),
        applies_to=_FOR,
        references=(ZHANG2022, ZHAPA2023, DASGUPTA2020),
        standalone=False,
        introduced_with=("bcbox", "cate"),
        used_by=("e2r",),
        implementation="mowl.training.graph.EdgeCorruption",
        notes=("BCBox (Section 4.4: 'randomly corrupting either the source or target node', 32 "
               "per edge), CatE (replacing the superconcept, 4 per morphism) and E2R (an "
               "individual of an assertion replaced, three per assertion). Differs from "
               "either-side-corruption and fixed-column-corruption only in where the random "
               "numbers come from: a seed of torch reproduces a run."),
    ),
    RegimeCard(
        key="known-row-mask",
        assumption="closed world over stated axioms",
        name="Leave known and reflexive corruptions out",
        kind=RegimeKind.TRAINING,
        summary=("A corruption that is a row of the training data of its form, or a reflexive "
                 "pair, is marked and left out of the means of the loss (or removed from the "
                 "negatives)."),
        components=("negative sampling",),
        applies_to=_FOR,
        references=(ZHANG2022, ZHAPA2023),
        standalone=False,
        introduced_with=("bcbox", "cate"),
        used_by=("e2r",),
        implementation="mowl.training.graph.KnownRowMask",
        notes=("BCBox masks them ('corruptions that hit a training edge or a self-loop'), "
               "CatE does with filter_negatives (its reference code does not), E2R drops the "
               "corruptions that are stated assertions. known-positive-filtering resamples "
               "instead."),
    ),
    RegimeCard(
        key="self-adversarial-bce",
        name="Binary cross entropy with self-adversarial negatives",
        kind=RegimeKind.TRAINING,
        summary=("The energy of every row plus the weighted -log(1 - exp(-E)) of its "
                 "corruptions, weighted by a softmax over the corruptions that the gradient "
                 "does not pass (RotatE), or all 1; the mean over the rows."),
        components=("loss combination",),
        applies_to=_FOR,
        references=(ZHANG2022,),
        standalone=False,
        introduced_with=("bcbox",),
        used_by=(),
        implementation="mowl.training.graph.SelfAdversarialBCE",
        notes=("Eq. 10 of Zhang et al. (2022) with the weights of Sun et al. (2019). The "
               "maths is the function bce_edge_loss of mowl.representation.library."
               "hierarchy_objectives; probabilistic-bce differs in the clamp and the form of "
               "the weights."),
    ),
    RegimeCard(
        key="grouped-margin",
        name="Margin loss of a row against its group of corruptions",
        kind=RegimeKind.TRAINING,
        summary=("The mean cost of the rows plus a hinge of margin gamma on the cost of their "
                 "corruptions, one hinge per corruption or one on their mean (the reference "
                 "code); marked corruptions are left out."),
        components=("loss combination",),
        applies_to=_FOR,
        references=(ZHAPA2023,),
        standalone=False,
        introduced_with=("cate",),
        used_by=(),
        implementation="mowl.training.graph.GroupedMarginLoss",
        notes=("Eq. 7 of CatE; the maths is margin_loss of mowl.representation.library."
               "hierarchy_objectives. pairwise-margin-ranking compares every positive with "
               "every negative of the step instead and has no mask."),
    ),
    RegimeCard(
        key="head-restricted-hinge",
        name="Interaction term on the rows of a set of heads",
        kind=RegimeKind.TRAINING,
        summary=("For the rows of a step whose first entity is in a given set, the mean cost "
                 "of the rows plus a hinge on one corruption whose second entity is drawn "
                 "from a given set of candidates; added to the loss of the step."),
        components=("loss combination", "negative sampling"),
        applies_to=_FOR,
        references=(ZHAPA2023,),
        standalone=False,
        introduced_with=("cate",),
        used_by=(),
        implementation="mowl.training.graph.HeadRestrictedHinge",
        notes=("The PPI term of the CatE code (src/cat_ppi_evaluator.py): proteins as heads, "
               "the elements 'exists r.p' as candidates. Not described in the paper; the "
               "published Table 4 was produced with it."),
    ),
    RegimeCard(
        key="validation-patience",
        name="Validation with patience and the best state restored",
        kind=RegimeKind.TRAINING,
        summary=("Every k epochs and after the last one a quantity of the model is "
                 "evaluated; the parameters are saved when it improves, training stops after "
                 "a number of evaluations without improvement, and the best parameters are "
                 "loaded at the end."),
        components=("model selection", "early stopping"),
        applies_to=_FOR,
        references=(ZHAPA2023,),
        standalone=False,
        introduced_with=("cate",),
        used_by=("hit", "ont"),
        implementation="mowl.training.graph.ValidationPatience",
        notes=("CatE: the raw mean reciprocal rank on the validation queries, patience 10. "
               "HiT and OnT: the best epoch by the validation MRR, which also chooses the "
               "weight lambda of the score (no stopping). validation-rank-selection keeps "
               "the best state in memory and does not stop."),
    ),
)
