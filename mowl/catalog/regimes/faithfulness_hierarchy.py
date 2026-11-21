"""Evaluation regime: the faithfulness hierarchy of Mashkova & Hoehndorf (2026)."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind

MASHKOVA2026 = Reference(
    "mashkova2026",
    "Mashkova, O., Hoehndorf, R. (2026). A hierarchy of faithfulness criteria for "
    "knowledge base completion. arXiv:2609.27863.",
    "https://arxiv.org/abs/2609.27863")

REGIME = RegimeCard(
    key="faithfulness-hierarchy",
    assumption="open world",
    task="axiom-ranking",
    target="logical status",
    name="Faithfulness hierarchy (entailed / contradictory / undetermined axioms)",
    kind=RegimeKind.EVALUATION,
    summary=(
        "Scores candidate axioms of each EL normal form whose logical status w.r.t. the "
        "training ontology is known: entailed (T+), contradictory (T-) and undetermined "
        "(T?). Tests four nested criteria of the score function: discrimination (L0), "
        "logical admissibility (L1), monotonic logical faithfulness (L2) and "
        "probabilistic logical faithfulness (L3, agreement with the relative model count "
        "P(a|O) = #(O + a) / #(O) over a bounded Herbrand domain)."),
    components=("test set", "candidate set", "metrics"),
    applies_to=("Any model that scores normal-form axioms (EL embedding models through "
                "their normal-form losses; any other model through a score function). "
                "Test-set generation needs a normalised EL++ ontology."),
    references=(MASHKOVA2026,),
    standalone=True,
    implementation="mowl.evaluation.faithfulness.FaithfulnessHierarchy",
    limitations=(
        "Test sets. mOWL decides the status of each candidate exactly by EL++ saturation "
        "(mowl.geometric.completion). 'Contradictory' needs a non-emptiness assumption "
        "for TBoxes without individuals (otherwise nothing is contradictory); mOWL "
        "assumes non-empty, unless provably empty, every concept name, every NF2 "
        "left-hand conjunction and every existential occurring in the ontology "
        "(witnesses='paper'), the concepts assumed non-empty by the paper's rule-based "
        "generators; 'names' and 'none' (classical consistency) are also available. "
        "Candidates are sampled uniformly from: pairs of classes (GCI0, GCI1-bot); "
        "attested conjunctions x classes (GCI1); classes x attested existentials (GCI2); "
        "attested existentials x classes (GCI3); trivial axioms, asserted axioms and "
        "axioms with an unsatisfiable left-hand side are excluded.",
        "Metrics. AUC+/- (Mann-Whitney, ties 1/2), Adm@k (per T+ axiom, fraction of the "
        "k best completions of its query not in T-; exact only if T- holds every "
        "contradictory candidate, or with exact_admissibility=True), FVR and Pair-FVR "
        "(exact counts, strict inequalities), rho_P (Spearman on T?), MRR and Hits@k "
        "(unfiltered tail ranking of T+, average ranks for ties), plus the four criteria "
        "as booleans on the finite test sets.",
        "rho_P needs relative model counts, which are #P-hard. mOWL's built-in exact "
        "counter (DPLL with component caching) handles small ontologies; Ganak through "
        "the optional pyganak package is supported but not tested in mOWL; the released "
        "datasets ship the authors' values.",
        "On the released Pizza data, mOWL's complete decision procedure finds some "
        "unknown_1000.owl axioms contradictory under the same non-emptiness assumption "
        "(the release's generators are rule based), and 22 test_neg.owl axioms of GCI0 "
        "are C SubClassOf owl:Thing.",
    ),
    notes=("Theory: the criteria form a strict chain PLF => MLF => admissibility => "
           "discrimination (Theorem 1, Appendix B); P(a|O) is 1 on Herbrand-entailed and "
           "0 on Herbrand-contradictory axioms (Proposition 1). The four KG baselines of "
           "the paper were scored on GCI0/GCI2 only. "
           "Benchmarks: faithfulness-pizza, faithfulness-goplus."),
)
