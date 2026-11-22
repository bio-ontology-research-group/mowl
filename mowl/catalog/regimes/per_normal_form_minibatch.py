"""Regime card: One mini-batch per normal form per step."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import (
    KULMANOV2019, MONDAL2021, JACKERMEIER2024, MASHKOVA2024NESY, AHARONI2024)

REGIME = RegimeCard(
    key='per-normal-form-minibatch',
    name='One mini-batch per normal form per step',
    kind=RegimeKind.TRAINING,
    summary=('Every optimisation step uses a mini-batch of each normal form (random with '
             'replacement, or consecutive batches that cycle) and the sum of their losses.'),
    components=('batching',),
    applies_to='EmbeddingELModel',
    references=(KULMANOV2019, MONDAL2021, JACKERMEIER2024, MASHKOVA2024NESY, AHARONI2024),
    standalone=False,
    introduced_with=('elem',),
    used_by=('emelpp', 'box2el', 'mcmultibox', 'e2r'),
    implementation='mowl.training.batching.PerNormalFormMiniBatches',
    notes=("ELEmbeddings, Algorithm 1, lines 18-22 ('Randomly sample a minibatch of size bs for "
           "each loss type'); EmEL++, Section 3.3; Box2EL reference code (get_data_batch, 512 "
           'axioms with replacement per normal form, one step per epoch); NeSy 2024/DELE '
           'reference code (cyclic consecutive batches, max_t ceil(n_t/b) steps per epoch, batch '
           "size 32,768); the E2R code (order 'replacement': batch_size rows drawn with replacement "
           "from the global torch generator; an epoch has max_k ceil(N_k/b) steps); Aharoni "
           "(2024), Section 7.1.3 (5 shuffled batches per epoch). mOWL's "
           'generic loop is full-batch; with this regime one mOWL epoch is one step.'),
)
