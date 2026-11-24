"""Regime card: Filtering of asserted positives from negatives."""

from mowl.catalog.card import RegimeCard, RegimeKind, Guarantee, GuaranteeKind
from mowl.catalog.regimes._regime_refs import KULMANOV2019, MONDAL2021, SINGH2026, LOURENCO2026

REGIME = RegimeCard(
    key='known-positive-filtering',
    name='Filtering of asserted positives from negatives',
    kind=RegimeKind.TRAINING,
    summary=('A sampled negative that is itself an asserted training axiom is re-sampled in its '
             'corrupted column (dropped if that fails). Entailed axioms are not filtered (compare'
             " 'dele-closure-filtered-negatives')."),
    components=('negative sampling',),
    applies_to="Any negative sampler following mOWL's negative layout",
    references=(KULMANOV2019, MONDAL2021, SINGH2026, LOURENCO2026),
    standalone=False,
    introduced_with=('elem',),
    used_by=('emelpp', 'elkem', 'boxlite'),
    implementation='mowl.training.negatives.KnownPositiveFilter',
    properties=(
        Guarantee(
            kind=GuaranteeKind.SOUNDNESS,
            statement=('No asserted training axiom of the same axiom type is returned as a '
                       'negative.'),
            source='by construction (KnownPositiveFilter)',
            condition='by construction',
            verified_in_mowl='tests/training/test_regime_negatives.py::TestFilters',
        ),
    ),
    limitations=(
        ('Only asserted axioms are removed; entailed axioms can still be sampled as negatives '
         '(DELE, Mashkova et al., Section 5).'),
    ),
    notes=("ELEmbeddings, Section 3.4 and EmEL++, Section 3.3 ('such that neither C' ⊑ ∃R.D nor C"
           " ⊑ ∃R.D' are asserted axioms'); ELK-Em, Section 4.2 (class assertions: 'filtering out"
           " observed training positives'); BoxLitE, Section 6 (corrupted role assertions 'not "
           "within the KB', for the test negatives)."),
)
