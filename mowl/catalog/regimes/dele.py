"""Training and evaluation regimes of DELE (Mashkova, Zhapa-Camacho, Hoehndorf) and of its
NeSy 2024 predecessor. DELE is not an embedding method: it is a way to train and evaluate
any geometric EL++ embedding using the deductive closure."""

from mowl.catalog.card import Guarantee, GuaranteeKind, Reference, RegimeCard, RegimeKind

MASHKOVA2024DELE = Reference(
    "mashkova2024dele",
    "Mashkova, O., Zhapa-Camacho, F., Hoehndorf, R. (2024). DELE: Deductive EL++ Embeddings "
    "for Knowledge Base Completion. Neurosymbolic Artificial Intelligence (NeSy 2024 special "
    "issue), arXiv:2411.01574.",
    "https://arxiv.org/abs/2411.01574")

MASHKOVA2024NESY = Reference(
    "mashkova2024nesy",
    "Mashkova, O., Zhapa-Camacho, F., Hoehndorf, R. (2024). Enhancing Geometric Ontology "
    "Embeddings for EL++ with Negative Sampling and Deductive Closure Filtering. In "
    "Neural-Symbolic Learning and Reasoning (NeSy 2024), Springer, 331-354. "
    "arXiv:2405.04868.",
    "https://arxiv.org/abs/2405.04868")

_USED_BY = ("elem", "elbe", "box2el")
_CLOSURE = ("deductive closure approximated by mowl.training.DeductiveClosure "
            "(DELE Algorithms 1-2): complete for GCI0 and GCI0-BOT, sound but incomplete "
            "for GCI1, GCI2, GCI3 and their BOT forms")

REGIMES = (
    RegimeCard(
        key="dele-all-normal-form-negatives",
        name="Negatives for all EL++ normal forms (DELE)",
        kind=RegimeKind.TRAINING,
        summary=(
            "Samples negatives for all seven EL++ normal forms, not only C ⊑ ∃R.D, by "
            "corrupting one class, and trains them with negative losses defined for each "
            "normal form and each geometric model (ELEmbeddings, ELBE, Box2EL)."),
        components=("negative sampling", "loss"),
        applies_to=("EmbeddingELModel whose module implements the DELE negative losses "
                    "(ELEmModule, ELBEModule, BoxSquaredELModule with "
                    "all_negative_losses=True)"),
        references=(MASHKOVA2024DELE, MASHKOVA2024NESY),
        standalone=True,
        used_by=_USED_BY,
        implementation="mowl.training.dele.AllNormalFormNegatives",
        properties=(
            Guarantee(
                GuaranteeKind.REGIME_PROPERTY,
                "Each negative loss is zero exactly when the geometric condition of DELE "
                "Eqs. 1-18 for the axiom not to hold is met (e.g. ELEm: the balls of A and B "
                "are at least the margin apart for A ⋢ B).",
                "mashkova2024dele, Section 4, Eqs. 1-18",
                condition="by construction",
                verified_in_mowl="tests/training/test_negative_losses.py"),
        ),
        limitations=(
            "Several negative losses only approximate non-entailment: ball or box "
            "separation is stronger than non-inclusion (DELE, Section 4.1).",
            "The ELBE/Box2EL GCI1-BOT negative loss uses the intersection offset "
            "|upper - lower| / 2, which is positive also for disjoint boxes, so a zero loss "
            "does not guarantee a non-empty intersection.",
        ),
        notes=(
            "The NeSy 2024 paper adds negative losses for GCI0-GCI3 to ELEmbeddings; DELE "
            "extends them to the BOT normal forms and to ELBE and Box2EL. The loss "
            "regularisation terms |‖f(x)‖ - 1| of the ELEm negative losses are applied by "
            "the module's global regularisation instead of per axiom. The reference code "
            "samples corrupting classes from the evaluation classes; mOWL from all named "
            "classes."),
    ),
    RegimeCard(
        key="dele-closure-filtered-negatives",
        name="Deductive-closure filtering of negatives (DELE)",
        kind=RegimeKind.TRAINING,
        summary=(
            "Removes from the sampled negatives every axiom that is in (an approximation "
            "of) the deductive closure of the training ontology, so that no entailed axiom "
            "is used as a negative."),
        components=("negative sampling",),
        applies_to="any EmbeddingELModel and any negative sampler (filter_negatives hook)",
        references=(MASHKOVA2024DELE, MASHKOVA2024NESY),
        standalone=True,
        used_by=_USED_BY,
        implementation="mowl.training.dele.ClosureFilteredNegatives",
        properties=(
            Guarantee(
                GuaranteeKind.REGIME_PROPERTY,
                "Sampled negatives are never in the computed deductive closure; since the "
                "closure rules are sound, every removed negative is entailed.",
                "mashkova2024dele, Section 5 and Appendix F (soundness of the rules)",
                condition=_CLOSURE,
                verified_in_mowl="tests/training/test_dele_regimes.py"),
        ),
        limitations=(
            "Entailed axioms outside the approximate closure (GCI1, GCI2, GCI3) can still "
            "be sampled as negatives.",
        ),
        notes=(
            "The papers delete entailed negatives; mOWL re-samples the corrupted column up "
            "to max_resample times and drops the rows still entailed afterwards "
            "(max_resample=0 reproduces deletion)."),
    ),
    RegimeCard(
        key="closure-positives",
        name="Deductive closure as training positives",
        kind=RegimeKind.TRAINING,
        summary=(
            "Adds entailed but not asserted GCI0, GCI2 (and optionally GCI3) axioms of the "
            "approximate deductive closure to the training data as positives."),
        components=("positives",),
        applies_to="any EmbeddingELModel",
        references=(MASHKOVA2024DELE,),
        standalone=False,
        implementation="mowl.training.dele.ClosurePositives",
        properties=(
            Guarantee(
                GuaranteeKind.REGIME_PROPERTY,
                "Every added positive is entailed by the training ontology.",
                "mashkova2024dele, Appendix F (soundness of the closure rules)",
                condition=_CLOSURE,
                verified_in_mowl="tests/training/test_dele_regimes.py"),
        ),
        notes=(
            "Not a contribution or experiment of DELE or the NeSy 2024 paper, which use "
            "the closure only to filter negatives and in evaluation. Provided by mOWL "
            "because the closure makes it cheap; the reference is to the closure algorithm."),
    ),
    RegimeCard(
        key="dele-evaluation",
        name="Deductive-closure-aware ranking evaluation (DELE)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "Ranks test axioms A ⊑ B or A ⊑ ∃r.B against all candidate classes, after "
            "removing test axioms entailed by the training ontology, and reports "
            "non-filtered and filtered (entailed candidates removed) Hits@n and macro / "
            "micro mean rank and AUC."),
        components=("test set", "candidate set", "filtering", "metrics"),
        applies_to="EL embedding models scoring gci0 or gci2 axioms",
        references=(MASHKOVA2024DELE, MASHKOVA2024NESY),
        standalone=True,
        used_by=_USED_BY,
        implementation="mowl.evaluation.deductive.DELEEvaluation",
        properties=(
            Guarantee(
                GuaranteeKind.REGIME_PROPERTY,
                "Filtered metrics never remove a candidate that is not entailed by the "
                "training and testing ontologies; the test axiom itself is never removed.",
                "mashkova2024dele, Section 6.2 and Appendix F",
                condition=_CLOSURE,
                verified_in_mowl="tests/evaluation/test_deductive.py"),
        ),
        limitations=(
            "The F metrics depend on the closure approximation: entailed GCI2 candidates "
            "outside it stay in the ranking.",
        ),
        notes=(
            "Ties get the average rank (scipy rankdata, as in the reference code). AUC is "
            "the rank-ROC of mowl.evaluation.base.compute_rank_roc in closed form, so that "
            "fractional ranks are allowed. Micro metrics average over the subclasses that "
            "occur in the test set (the paper says 'all classes in the signature'; the "
            "reference code averages over test subclasses). The reference code filters by "
            "multiplying scores by 1e6, which also affects the test axiom and zero "
            "scores; mOWL removes filtered candidates from the ranking instead. NeSy 2024 "
            "filters with the training axioms only: filter_with='train'."),
    ),
)
