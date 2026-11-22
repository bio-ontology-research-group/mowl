How ontology embeddings are trained and tested
==============================================

A paper on an ontology embedding method describes a *representation* (balls, boxes,
cones, a language model, ...) together with a *training procedure* and an *evaluation
procedure*. mOWL keeps the three apart (see :doc:`index`). This page classifies, from the
papers, how the methods are trained and how they are tested, names the procedures that
recur, and points out where papers that look comparable use different procedures. The
named procedures are the training and evaluation *regimes* of the catalogue; most of them
were never published on their own but appear in an experiments section, an appendix or
only in the reference code.

Every row below cites the section that was read. "Code" means the authors' reference
implementation was consulted because the paper does not say; "not stated" means the
paper (and, where checked, the code) does not say.


Training
--------

Dimensions
^^^^^^^^^^

* **Positives**: asserted (normalised) axioms only; plus the deductive closure; plus a
  rule- or reasoner-materialised structure; or sentences of a corpus.
* **Negatives**: none; uniform corruption of a fixed column, of either side, or of each
  side separately; type-constrained; filtered against asserted positives or against the
  deductive closure; hard (siblings); provably false (from disjointness); self-adversarially
  weighted; not sampled but expressed as a regulariser.
* **Score combination**: sum of (margin) losses; squared error to targets; pairwise
  ranking; log-sigmoid; BPR; binary cross-entropy; log-sum-exp contrastive; Pearson
  correlation.
* **Regularisation**, **initialisation**, **optimiser and schedule**, **batching**
  (full batch, one mini-batch per normal form per step, mini-batches over all axioms) and
  **model selection** (validation loss, validation rank metric, grid search only).

Table
^^^^^

.. list-table:: How the papers train
   :header-rows: 1
   :widths: 12 14 22 14 14 24

   * - Paper (sections)
     - Positives
     - Negatives
     - Score combination, regularisation
     - Optimiser, schedule, batching
     - Model selection, other
   * - ELEmbeddings (Kulmanov et al. 2019, §3.2-3.4, Alg. 1, §4.2)
     - asserted normal forms; ABox rewritten to nominals
     - NF3 :math:`C \sqsubseteq \exists R.D` only; replace :math:`C` *or* :math:`D`;
       never an asserted axiom
     - sum of hinge losses with margin :math:`\gamma`; every loss contains
       :math:`|\|c\| - 1|` (centres on the unit sphere); init uniform(0, 1)
     - Adam (TensorFlow); one mini-batch per loss type per update (Alg. 1)
     - grid over dimension and :math:`\gamma` on validation PPI metrics; no checkpoint
       selection stated
   * - EmEL++ (Mondal et al. 2021, §3.3, §4.3)
     - asserted normal forms, role inclusions and chains
     - as ELEmbeddings
     - sum; norm terms; penalty on negative radii
     - Adam; mini-batches per loss; "till saturation or a fixed number of epochs"
     - grid (dim, :math:`\gamma`) on a 20% validation split
   * - EmEL(var) (Mohapatra et al. 2021, §5)
     - as EmEL++
     - not stated beyond EmEL++
     - as EmEL++
     - not stated
     - as EmEL++
   * - ELBE (Peng et al. 2022, §4, §5.1; code)
     - asserted normal forms
     - NF3 (Eq. 10); side not stated; code draws from a precomputed pool of NF3
       negatives
     - paper: sum of norms; code: squared error per axiom, target 0 for positives and 2
       for negatives
     - Adam, lr grid :math:`\{10^{-2}, \dots, 5\cdot10^{-4}\}`
     - grid (dim, margin)
   * - BoxEL (Xiong et al. 2022, §4.5, §5.2-5.3)
     - asserted normal forms
     - role assertions: head *or* tail; NF1: :math:`C \not\sqsubseteq D'` or
       :math:`C' \not\sqsubseteq D`, loss weighted by :math:`\phi = 0.05`
     - sum; unit-cube regulariser (Eq. 12); softplus volume with temperature
     - Adam
     - grid (dim, :math:`\gamma`); mean of 10 seeds
   * - Box²EL (Jackermeier et al. 2024, §3.2, §4, App. E, H; code)
     - asserted normal forms and role inclusions
     - NF3 and role assertions; paper: replace :math:`C` or :math:`D`; **code: one
       tail- and one head-corrupted copy of every axiom**; :math:`\omega \in \{1..5\}`
       sets, resampled every epoch
     - squared per-axiom losses; negatives :math:`(\delta - \mu)^2`; :math:`L_2` on bump
       vectors (:math:`\lambda`)
     - Adam, up to 10,000 epochs (code default 5,000); lr :math:`\times 0.1` after 2,000
       epochs for some settings; code: 512 axioms per normal form with replacement, one
       step per epoch
     - median rank of the first 1,000 validation NF1 axioms every 100 epochs (code);
       grid on validation; mean of 5 seeds
   * - TransBox (Yang et al. 2025, §4.3, §6.2)
     - asserted normal forms plus semantic and intersection enhancements
     - NF3: replace :math:`A` with :math:`A'` *and* :math:`B` with :math:`B'`
     - as Box²EL ("re-implementation of [Box²EL]"); :math:`\lambda = 1`
     - Adam, 5,000 epochs
     - grid on validation; mean of 10 runs
   * - ELK-Em (Singh et al. 2026, §4.2, §5.1-5.2)
     - asserted normal forms (some bottom forms rerouted)
     - GCI0: replace :math:`D`, :math:`k_{neg}` per positive, unfiltered; GCI2:
       :math:`C` or :math:`D` with equal probability; class assertions: class
       replaced, training positives filtered
     - sum of per-axiom losses plus Pearson-correlation contrastive term (TBox only);
       ABox :math:`\mathrm{ReLU}(\delta - \ell)`; lexical regularisation
     - Adam; lr 0.01, :math:`\times 0.1` after 5,000 epochs (§5.1) or constant then
       cosine (§5.2); batch 1,024
     - median rank on validation; mean of 5 seeds
   * - NeSy 2024 paper (Mashkova et al., §4.3-4.5; code)
     - asserted axioms
     - all normal forms, one column each; random or deductive-closure filtered; code:
       GO classes and proteins from separate pools
     - sum weighted by normal-form frequency (Eq. 4; code: share of mini-batches);
       LeakyReLU relaxation; relaxed or strict norm constraint
     - Adam, ReduceLROnPlateau (patience 10), early stopping (20); batch 32,768; code:
       cyclic consecutive batches per normal form
     - validation loss (code: GCI2); one fixed seed
   * - DELE (Mashkova et al., §4-6)
     - asserted axioms (closure only for filtering)
     - as the NeSy paper, negative losses for all seven normal forms
     - sum (Eq. 23)
     - as the NeSy paper; 2,000 or 800 epochs
     - validation loss and early stopping
   * - FALCON (Tang et al. 2022, §3, §5.1)
     - ABox and TBox axioms; anonymous individuals resampled before every step
     - uniform, per ABox axiom (4, 8 or 16); side not stated
     - BPR for the ABox; fuzzy TBox loss
     - Adam, lr grid; batch sizes per axiom type
     - grid search
   * - CatE (Zhapa-Camacho and Hoehndorf, §3.2, §4.1, §4.5)
     - edges of a concept lattice built from the ALC axioms, optionally saturated
       (rules S1, transitive closure "Tr")
     - every relationship: replace :math:`B` uniformly (1, 2 or 4)
     - order score plus hinge on negatives (Eq. 7)
     - Adam with cyclical learning rate; batch 8,192-32,768
     - grid on validation
   * - RegD (Yang and Chen, §3.3, §5)
     - DAGs: only basic edges (transitive reduction); ontologies: those of the wrapped
       model
     - random replacement
     - log-sum-exp contrastive loss of shadow cones (Eq. 9)
     - not stated in the sections read
     - threshold on the evaluation set
   * - OnT (Yang et al. 2025, §3, §5)
     - :math:`C \sqsubseteq D`, :math:`\exists r.D`, :math:`C \sqcap D` of its own
       normalisation, with labels
     - one random negative per axiom
     - triplet (hierarchical contrastive) + centripetal + role and conjunction losses
     - pretrained language model, lr :math:`10^{-5}`, 1 epoch
     - score weight :math:`\lambda` tuned on validation
   * - HiT (He et al. 2024, §2.3, §3, §4)
     - one-hop (asserted) subsumptions
     - 10 per positive, random or sibling ("hard"); never in :math:`E \cup T`
     - triplet loss (Eq. 3) + centripetal loss
     - pretrained language model
     - threshold on validation
   * - Aharoni (2024, §7.1)
     - asserted normal forms (Box²EL code base)
     - as Box²EL code
     - Monte-Carlo volume losses (20,000 points)
     - 5 epochs of 5 shuffled batches
     - none (fixed epochs)
   * - BoxLitE (Lourenço et al. 2026, §5)
     - DL-Lite KB
     - none sampled: convex regularisation on existential concepts
     - second-order cone program
     - MOSEK
     - not stated
   * - E2R (Garg et al. 2019, §3-4)
     - TBox and ABox memberships
     - "tiny fraction" of corrupted memberships, one entity replaced
     - negated membership loss as regulariser
     - Adam
     - not stated
   * - Query2Box, ConE, AConE (Ren et al. 2020, Eq. 4; Zhang et al. 2021, §4; He et al.
       2024, Eq. 4)
     - training queries and answers
     - :math:`k` random entities
     - log-sigmoid with uniform weights :math:`1/k`
     - Adam; grid over lr, batch, :math:`k`, margin
     - not checked
   * - Octagons (Charpenay and Schockaert 2024, §5)
     - KG triples
     - :math:`|N|` random
     - log-sigmoid, self-adversarial weights
     - Adam, lr :math:`10^{-2}` or :math:`10^{-3}`
     - not stated
   * - Binary-code boxes (Zhang et al. 2022, §4.4)
     - graph edges
     - 32 per edge, source *or* target corrupted, self-adversarial
     - binary cross-entropy (Eq. 10); lasso on binary codes
     - mini-batches
     - not stated
   * - Gumbel boxes (Dasgupta et al. 2020, §5)
     - WordNet: transitive reduction
     - not stated for WordNet
     - cross-entropy / KL to conditional probabilities
     - not stated
     - not stated
   * - Order embeddings (Vendrov et al. 2016, §2-4)
     - transitive closure
     - random pairs
     - positives' penalty plus hinge on negatives
     - not checked
     - not checked
   * - OWL2Vec*, OPA2Vec, Onto2Vec, DL2Vec (survey of Chen et al. 2025, §III; mOWL's
       graph and syntactic models)
     - random walks over a graph projection, or axioms and annotations as sentences
     - word2vec's internal negative sampling
     - skip-gram / CBOW objective
     - word2vec
     - downstream classifier

Training regimes
^^^^^^^^^^^^^^^^

The recurring, representation-independent procedures, with their catalogue keys
(regimes owned by other parts of mOWL are in parentheses):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Component
     - Used by
   * - ``fixed-column-corruption``
     - negatives
     - mOWL default, CatE, ELK-Em (GCI0), DELE (one column per normal form)
   * - ``either-side-corruption``
     - negatives
     - ELEmbeddings, EmEL++, BoxEL, Box²EL (paper), ELK-Em (GCI2), binary-code boxes
   * - ``per-column-corruption``
     - negatives
     - Box²EL (code), TransBox
   * - ``type-constrained-corruption``
     - negatives
     - NeSy 2024 / DELE (code), PPI examples of mOWL (one type)
   * - ``known-positive-filtering``
     - negatives
     - ELEmbeddings, EmEL++, ELK-Em (ABox)
   * - (``dele-closure-filtered-negatives``)
     - negatives
     - NeSy 2024, DELE
   * - ``sibling-hard-negatives``
     - negatives
     - HiT
   * - ``contradiction-negatives``
     - negatives
     - Mashkova and Hoehndorf 2026 (§6.3)
   * - ``existential-concept-negatives`` (not implemented)
     - negatives
     - BoxLitE
   * - (``dele-all-normal-form-negatives``)
     - negatives, loss
     - NeSy 2024, DELE
   * - ``transitive-reduction-positives``
     - positives
     - RegD, HiT, Gumbel boxes
   * - ``transitive-closure-positives``
     - positives
     - order embeddings, CatE ("Tr")
   * - (``closure-positives``)
     - positives
     - mOWL extension
   * - ``loss-sum``
     - loss
     - ELEmbeddings, EmEL++, DELE, BoxEL (:math:`\phi`), Box²EL and TransBox (squares in
       the module)
   * - ``mse-to-target``
     - loss
     - ELBE (code), mOWL's ELBE (variant)
   * - ``pairwise-margin-ranking``
     - loss
     - HiT, OnT
   * - ``logsigmoid-negative-sampling``
     - loss
     - Query2Box, ConE, AConE, octagons (self-adversarial)
   * - ``bpr-loss``
     - loss
     - FALCON
   * - ``probabilistic-bce``
     - loss
     - binary-code boxes, Gumbel boxes
   * - ``logsumexp-contrastive``
     - loss
     - RegD (paper, Eq. 10)
   * - ``softmax-energy-contrastive``
     - loss
     - RegD (code: softmax over the energies of one positive and its 10 negatives, on the
       task's normal form only)
   * - ``pearson-correlation-loss``
     - loss
     - ELK-Em
   * - ``normal-form-frequency-weighting``
     - loss
     - NeSy 2024
   * - ``step-lr-decay``, ``cosine-lr-decay``, ``cyclic-lr``, ``reduce-lr-on-plateau``
     - schedule
     - Box²EL and ELK-Em; ELK-Em; CatE; NeSy 2024 and DELE
   * - ``early-stopping``
     - schedule
     - NeSy 2024, DELE
   * - ``per-normal-form-minibatch``
     - batching
     - ELEmbeddings, EmEL++, Box²EL (code), DELE (code), Aharoni, RegD (code)
   * - ``validation-loss-selection``, ``validation-rank-selection``, ``validation-f1-selection``
     - model selection
     - mOWL default, DELE; Box²EL, ELK-Em; RegD (code, the task's normal form)

Training regimes are :class:`~mowl.training.TrainingRegime` objects in
:mod:`mowl.training.negatives`, :mod:`mowl.training.positives`,
:mod:`mowl.training.losses`, :mod:`mowl.training.schedules`,
:mod:`mowl.training.batching` and :mod:`mowl.training.selection`. They compose:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ELEmbeddings
   from mowl.training import ComposedRegime
   from mowl.training.negatives import EitherSideCorruption, KnownPositiveFilter
   from mowl.training.schedules import StepDecay

   model = ELEmbeddings(FamilyDataset(), embed_dim=4, batch_size=16)
   model.training_regime = ComposedRegime(
       StepDecay(milestones=[2]),                 # provides the optimiser
       EitherSideCorruption(num_negatives=2),     # ELEmbeddings' negatives (Section 3.4)
       KnownPositiveFilter())                     # ... never an asserted axiom
   model.train(epochs=3, validate_every=10)

Some procedures are representation-specific and are not regimes: the norm constraints
of ELEmbeddings, the unit-cube regulariser of BoxEL, the bump regulariser of Box²EL, the
LeakyReLU relaxation of the NeSy 2024 paper, the anonymous-individual sampling of FALCON
and the centripetal losses of HiT and OnT live in the modules.


Evaluation
----------

Dimensions
^^^^^^^^^^

* **Task**: axiom prediction on a random split; approximate deductive inference
  (train on everything, test on entailed axioms); link or PPI prediction; membership
  prediction; query answering; classification with a threshold; model checking;
  faithfulness.
* **Test set**: which axioms, which normal forms, how leakage into training is avoided.
* **Ranked side and candidates**: subclass or superclass, head or tail; all classes, all
  named classes except trivial answers, proteins only, a sample.
* **Filtering**: none (raw); training (and validation) positives; the deductive closure.
* **Score**: the training loss, the loss without margin, a centre distance, a volume
  ratio, a probability.
* **Metrics and aggregation**: Hits@k, MR, MRR, median and percentile ranks, three
  different ROC AUCs, F1; averaged over test axioms ("macro" in DELE), over head entities
  ("micro" in DELE), per normal form, per query type, over seeds.

Table
^^^^^

.. list-table:: How the papers test
   :header-rows: 1
   :widths: 12 18 18 12 14 26

   * - Paper (sections)
     - Task, test set
     - Ranked side, candidates
     - Filtering
     - Score
     - Metrics, aggregation
   * - ELEmbeddings (§4.2)
     - PPI (yeast/human, STRING :math:`\ge 700`); 80/10/10 split of interaction pairs
     - :math:`P_2` given :math:`P_1` (tail), all training proteins
     - raw and filtered (training and validation interactions)
     - Eq. 21: NF3 loss with :math:`\gamma`, without norm terms
     - H@10, H@100, MR, AUC
   * - EmEL++ / EmEL(var) (§4.3-4.5 / §5)
     - subsumption; 70/20/10 split of NF1; "inferences" (ELK on the training part)
     - **subclass** :math:`C` given :math:`D`, all other classes
     - raw
     - distance from :math:`D`
     - H@1/10/100, median, 90th percentile rank; inclusion accuracy (Table 4)
   * - ELBE (§5.2-5.3)
     - PPI as ELEmbeddings; synthetic GO intersection equivalences
     - PPI: :math:`P_1` given :math:`P_2` (**head**); :math:`E` for :math:`C \sqcap D`
     - raw and filtered
     - Eq. 23 (loss without margin); intersection-centre distance
     - H@10/100, MR, AUC; H@1/3/10, MR
   * - BoxEL (§5.2-5.3)
     - subsumption 70/20/10 (GO, GALEN, Anatomy); PPI
     - "rank the probability of C subsumed by all concepts" (side ambiguous)
     - raw (subsumption), raw and filtered (PPI)
     - volume-based probability; PPI Eq. 14
     - H@10/100, MR, AUC; inclusion accuracy (Table 3); mean of 10 seeds
   * - Box²EL (§4.2-4.4, App. F, G)
     - prediction: 80/10/10 of all normal forms, complex concepts included; deductive:
       train on all, test on ELK-entailed non-asserted NF1 (10% validation); PPI
     - NF1 :math:`D`, NF2 :math:`E`, NF3 :math:`C` (the subclass), NF4 :math:`D`, all concepts
       (trivial answers removed); PPI :math:`P_1` (head) among proteins
     - raw; PPI raw and filtered (training)
     - centre distance (Appendix F), not the loss
     - H@1/10/100, Med, MRR, MR, AUC per normal form and combined; mean of 5 seeds
   * - TransBox (§6.2, App. D-F)
     - complex axioms generated by forgetting (LETHE); normalised axioms as Box²EL
     - per side (``?C ⊑ *``, ``* ⊑ ?A``); normalised as Box²EL
     - raw
     - centre distance with emptiness masks; NF2 score 0 if :math:`A \cap B = \emptyset`
     - as Box²EL; mean of 10 runs
   * - ELK-Em (§5.1-5.2)
     - Box²EL deductive protocol (90/10 test/validation); CAFA5 proteins held out
     - :math:`D` given :math:`C`, full signature; GO terms
     - raw
     - :math:`\exp(-\ell_0)` (loss with margin 0); baselines' numbers copied (centre
       distance)
     - Box²EL metrics; protein-centric F1 at a validation threshold
   * - NeSy 2024 / DELE (§4.5 / §6.1-6.2)
     - PPI and protein function (mOWL datasets); FoodOn, GALEN 90:5:5, leaked and
       entailed test axioms removed
     - tail :math:`D` (or :math:`B` of :math:`\exists r.B`), all classes or a type
     - raw and filtered by training axioms **and the deductive closure**
     - the loss
     - H@10/100; "macro" MR/AUC over test axioms, "micro" MR/AUC averaged over head
       classes; ties averaged (NeSy 2024 code)
   * - FALCON (§5.1-5.2)
     - Pizza: HermiT entailments vs. axioms neither asserted nor entailed, with injected
       inconsistencies; HPO gene-gene; yeast PPI (ELEmbeddings split)
     - --; filtered ranking (HPO); PPI as ELEmbeddings
     - filtered (HPO); raw and filtered (PPI)
     - membership degree
     - MAE, AUC, AUPR, Fmax; H@10/100, MR, AUC
   * - CatE (§4.1-4.4)
     - ORE1 :math:`C(a)`; GO/FoodOn 70/10/20 of :math:`C \sqsubseteq D`; PPI
     - class of :math:`C(a)`; :math:`D` among named classes
     - filtered only (training; plus deductive closure for :math:`C(a)`)
     - order loss
     - MR, MRR, H@3/10/100, AUC
   * - RegD (§5)
     - DAG: 5% of the non-basic edges; ontologies: 1,000 sampled ELK-inferred
       subsumptions (inference) or :math:`\exists r.B \sqsubseteq ?A` of an 80/10/10
       split (prediction)
     - classification of pairs, child replaced 10 times
     - --
     - energy below a threshold tuned on validation
     - F1
   * - OnT (§5)
     - own normalisation; prediction 80/10/10; inference: all ELK-inferred NF1
     - all atomic concepts
     - not stated
     - Eq. 8 (distance + :math:`\lambda` norm difference, :math:`\lambda` on validation)
     - H@1/10/100, MRR, MR
   * - HiT (§4)
     - WordNet multi-hop (train one-hop, test transitive) and mixed-hop; transfer
     - pairs with 10 negatives (random or siblings)
     - --
     - hyperbolic score with validation threshold
     - precision, recall, F1
   * - Aharoni (§7.1.2)
     - Box²EL splits (GALEN, GO)
     - as Box²EL
     - raw
     - centre distance; Monte-Carlo membership for multiboxes
     - H@10/100, mean and median rank
   * - BoxLitE (§6)
     - Family F v1-4: inferred role assertions split 20/80 validation/test
     - :math:`a` or :math:`b` of :math:`R(a, b)`, all individuals
     - filtered
     - negative loss
     - MRR, H@1/3/10
   * - E2R (§5)
     - FB15K, WN18; LUBM1U memberships
     - entities of membership queries
     - filtered
     - membership loss
     - MR, MRR, H@1/10 averaged per query
   * - Query2Box / ConE / AConE
     - complex queries; only non-trivial answers
     - all entities
     - filtered
     - query distance
     - MRR, H@k per query type
   * - Octagons (§6)
     - FB15k-237, WN18RR
     - entities
     - not stated in the text read
     - octagon distance
     - H@1/3/10, MRR
   * - Binary-code boxes (§5)
     - graph link prediction
     - test edges vs. uniform or corrupted negatives
     - --
     - probability
     - AUC (two negative strategies)
   * - Gumbel boxes (§5.3)
     - WordNet transitive closure
     - pairs, 1:10 negatives
     - --
     - conditional probability
     - F1; MRR on trees
   * - Mashkova and Hoehndorf 2026 (§5-6)
     - reasoner-generated entailed, contradictory and undetermined axioms per normal
       form (Pizza, GO-plus slice)
     - --
     - --
     - model score
     - AUC±, FVR, Pair-FVR, MRR, H@10
   * - Survey (Chen et al. 2025, §IV-A, §VI)
     - "prediction" vs. "approximate deductive inference"; candidates "quite flexible"
     - sub-concept given, super-concepts ranked
     - mOWL demo: filtered (training)
     - --
     - MR, MRR, H@k, AUC

Evaluation regimes
^^^^^^^^^^^^^^^^^^

Implemented in :mod:`mowl.evaluation` (keys of regimes written for the benchmarks and
for DELE and the faithfulness hierarchy in parentheses):

* (``box2el-normal-form-ranking``), (``inferred-subsumption-ranking``),
  (``ppi-ranking``), (``emelpp-subsumee-ranking``), (``cate-filtered-ranking``),
  (``dele-evaluation``), (``regd-sampled-f1``), (``exact-model-check``),
  (``faithfulness-hierarchy``), (``modelness``);
* ``region-inclusion-accuracy`` (EmEL++, BoxEL);
* scoring rules, a component of every ranking regime: ``loss-scoring``,
  ``centre-distance-scoring`` and ``margin-free-loss-scoring`` (ELK-Em).

Catalogued, not implemented: ``intersection-equivalence-ranking`` (ELBE),
``complex-axiom-prediction`` (TransBox), ``paraconsistent-entailment-classification``
(FALCON), ``protein-centric-threshold-f1`` (ELK-Em), ``hierarchy-threshold-f1`` (HiT,
Gumbel boxes), ``sampled-negative-auc`` (binary-code boxes),
``kg-filtered-link-prediction`` (BoxLitE, octagons, E2R) and ``query-answering``
(Query2Box, ConE, AConE).


Where comparable-looking numbers differ in regime
-------------------------------------------------

* **Which side is ranked.** For :math:`C \sqsubseteq D`, EmEL++ and EmEL(var) rank the
  subclass :math:`C`; Box²EL, ELK-Em, CatE and DELE rank the superclass :math:`D`. For
  PPI, ELEmbeddings ranks :math:`P_2`, ELBE and Box²EL rank :math:`P_1`. For
  :math:`C \sqsubseteq \exists r.D` Box²EL ranks :math:`C`, DELE ranks :math:`D`.
* **Score.** Box²EL, TransBox, EmEL++ and Aharoni rank by a centre distance; ELEmbeddings,
  DELE, CatE and mOWL's evaluators by the loss; ELK-Em by the margin-free loss. ELK-Em's
  Table 3 compares its own loss-based ranks with Box²EL's centre-distance numbers. With a
  positive margin the loss produces ties ("ranks thousands of axioms at the same rank",
  NeSy 2024, §5), so the tie rule matters: Box²EL's code breaks ties by position, DELE's
  averages them.
* **Filtering.** Box²EL and ELK-Em report raw ranks (ELK-Em keeps them "for
  comparability"), CatE filtered only, DELE filters by the deductive closure, PPI papers
  report both. Raw and filtered H@10 differ by a factor of two or more on yeast PPI
  (ELBE Table 3: 0.11 vs. 0.26).
* **Which column is corrupted.** Box²EL's paper says "C or D", its code corrupts both;
  TransBox copies the code; mOWL's BoxSquaredEL corrupts only :math:`D` and does not use
  its ``num_negs``. DELE corrupts a single column per normal form. Box²EL's Table 9 shows
  that the number of negatives alone moves median rank from 7351 (none) to about 630.
* **Negatives for one normal form or all.** ELEmbeddings, ELBE and Box²EL sample
  negatives only for NF3; CatE and DELE for all relationships or normal forms and report
  that the former collapse on the other normal forms.
* **Test sets called "inferences".** EmEL++'s inferences are ELK inferences of the
  training part, which Box²EL found to overlap with the training data (footnote 5);
  Box²EL uses entailed but *not asserted* NF1 axioms; OnT uses all inferred NF1 axioms of
  its own normalisation; RegD samples 1,000 of them and classifies with a threshold.
* **Candidate sets.** All classes (Box²EL), all named classes except trivial answers
  (Box²EL removes :math:`C` from :math:`C \sqsubseteq ?D`), only proteins (PPI), a
  per-type set (DELE), about 50 candidates (OWL2Vec*'s original setting, as noted by OnT,
  §5.2).
* **AUC.** Three different ROC AUCs are in use: Box²EL's trapezoid over false positive
  rates, the rank-curve integral of the ELEmbeddings/Box²EL PPI code and mOWL's Riemann
  sum (:mod:`mowl.evaluation.ranking_metrics`).
* **Aggregation.** DELE's "micro" metrics average per head class, its "macro" ones over
  axioms (the opposite of the usual naming); Box²EL reports per normal form and combined;
  E2R averages per query; Box²EL and ELK-Em average 5 seeds, BoxEL and TransBox 10,
  DELE uses one seed.
* **Model selection.** mOWL's default loop and DELE keep the parameters with the lowest
  validation loss of one normal form; Box²EL and ELK-Em the best validation median rank.
  An embedding in which every region grows has a low positive validation loss but a bad
  rank.
* **Epochs and batching.** In the Box²EL and ELK-Em code an "epoch" is one step on one
  mini-batch per normal form; in DELE's code it is a pass over all batches; in mOWL's
  generic loop it is one full-batch step. Epoch counts are therefore not comparable.


Missing hooks
-------------

The :class:`~mowl.training.TrainingRegime` hooks express negatives, filters, positives,
loss combination, per-epoch schedules and the optimiser. Some procedures above need more:

* **early stopping** needs a hook that can end training (``should_stop(epoch)``);
* **model selection** needs a hook after the last epoch (``on_train_end``) to evaluate
  the final parameters and restore the best ones; :class:`~mowl.training.selection.ValidationSelection`
  asks the user to call ``finalize()`` instead;
* **several optimisation steps per epoch** need a batch-level hook (``batches(gci_name,
  dataset)`` and a step per batch); :class:`~mowl.training.batching.PerNormalFormMiniBatches`
  maps one step to one epoch;
* **score-based losses** (pairwise, log-sigmoid, BPR, BCE, Pearson) need the ordinary
  scores of the negatives; the loop passes the module's *negative* losses, so these
  regimes record the negatives in ``filter_negatives`` and score them again. A
  ``loss(gci_name, pos_scores, neg_scores, neg_data)`` signature, or a ``score`` hook,
  would avoid the second forward pass.


General parts that named methods are made of
--------------------------------------------

The named classes (HiT, OnT, EIKE, BoxLitE, the query models, MultiBoxEL, PrismEL) name
their parts and hand rows to them; the arithmetic lives in general modules that any other
method can use:

* :mod:`mowl.evaluation.text_ranking`: encoding texts in batches and with a cache, embedding
  distinct pairs once, scoring candidates in chunks, choosing a score parameter on a grid by
  mean reciprocal rank (the evaluation side of every text-encoder method).
* :mod:`mowl.evaluation.query_scoring`: scoring queries (a union is the best branch of the
  disjunctive normal form) and candidate axioms against a query geometry.
* :mod:`mowl.evaluation.signature_costs`: costs of class-assertion rows for a model that has
  its own vocabulary; rows outside its signature cost infinity.
* :mod:`mowl.representation.library.text_encoders` and
  :mod:`mowl.representation.library.poincare_ball`: text encoders and the Poincare-ball
  operations.
* :mod:`mowl.representation.library.vector_tables`: named vectors (PCA reduction, saving,
  loading).
* :mod:`mowl.representation.library.structural_tables`: definition tables and targets of the
  structural reading.
* :mod:`mowl.training.query_sampler`, :mod:`mowl.training.query_structures` and
  :mod:`mowl.projection.query_graph`: the query sampler, the query structures and the ontology
  as a graph of entities and relations.

.. testcode::

   import torch
   from mowl.evaluation.text_ranking import best_on_grid, candidate_scores

   # three queries, four candidates; the score has one parameter, tried on a grid
   table = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.0, 0.8, 0.1, 0.0], [0.0, 0.0, 0.2, 0.7]])
   targets = torch.tensor([0, 1, 3])
   mrr, value = best_on_grid(lambda w, a, b: w * table[a:b], targets, (0.5, 1.0))
   assert mrr == 1.0 and value == 0.5
   both = candidate_scores(lambda q, c: q @ c.T, table, table[:3], chunk_size=2)
   assert both.shape == (3, 3)
