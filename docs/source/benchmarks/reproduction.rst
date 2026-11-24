Reproduction status
===================

This page records, for every method with a recipe (see :doc:`index`), whether mOWL
reproduces the numbers printed in the publication. A method counts as *reproduced* when a
run with its recipe,

.. code-block:: bash

   python -m mowl.benchmarks run <benchmark> --method <key> --recipe

gives the published numbers within seed noise. All runs were made on GPUs in November 2026
with the recipes of this version; "ours / published" is given throughout. Unless a seed
count is stated, the number is the mean of 3 seeds.

The status values are:

* **reproduced**: all reported metrics agree within seed noise;
* **partly**: some benchmarks or metrics agree, others do not;
* **not reproduced**: the numbers differ and the reason is stated;
* **no benchmark**: the method is implemented and tested, but no published benchmark can
  be run against it.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 16 22 14 48

   * - Method
     - Benchmarks
     - Status
     - Remarks
   * - Box\ :sup:`2`\ EL
     - ``box2el-*``, ``transbox-*-prediction``
     - reproduced
     - 1 seed, trained with the role inclusions and role chains, as in the paper.
       Largest gaps: GALEN prediction 0.31 / 0.35 and ANATOMY prediction 0.66 / 0.70
       (Hits\@100).
   * - TransBox
     - ``transbox-*-prediction``
     - reproduced
     - All normal forms; the paper averages 10 seeds.
   * - ELEm, EmEL++, ELBE (baselines of the Box\ :sup:`2`\ EL paper)
     - ``box2el-*``
     - reproduced
     - With ``Box2ELCodeTraining(baseline_code=True)``.
   * - BoxEL
     - ``box2el-*``
     - partly
     - Reproduced except where the published number is produced by a ranking bug of
       the reference code for ``∃r.C ⊑ D``.
   * - OnT
     - ``ont-*``
     - reproduced
     - All six benchmarks.
   * - HiT
     - ``ont-*``
     - partly
     - Four of six. GO inference: the paper reports a mean rank of 44,253, mOWL 882.
   * - CatE
     - ``cate-*``
     - partly
     - FoodOn and GO reproduced. ORE1 cannot be reproduced exactly from the released
       data. Yeast PPI not finished.
   * - DELE (ELEm, ELBE, Box\ :sup:`2`\ EL; base, +l, +l+n)
     - ``dele-*``
     - partly
     - Most cells of FoodOn, GALEN and yeast PPI within a few percent; yeast function
       only for Box\ :sup:`2`\ EL.
   * - RegD
     - ``regd-*``
     - partly
     - About half of 48 cells within 0.04 F1. The prediction-task numbers of the paper
       depend on a bug in its evaluation code. The authors' released code, run with its
       own sweep, reproduces the paper's ELBE, ELEm and RegD-with-balls numbers but not
       RegD with boxes (0.39 against 0.63 on ANATOMY inference); mOWL's RegD with boxes is
       at or above the released code.
   * - Concept2Box
     - ``concept2box-*``
     - not reproduced
     - MRR 0.02 / 0.30 / 0.73 (instance view / ontology view / concept linking) against
       0.51 / 0.65 / 0.87. The instance-view numbers of the paper cannot come from the
       released data: its TransE row is the "TransE (all)" row of the JOIE paper, computed
       on the larger, unreleased DB111K-174. On the released data plain TransE reaches MRR
       0.09 to 0.16 against 0.32.
   * - EIKE
     - ``yago39k-*``, ``m-yago39k-*``
     - not reproduced
     - F1 of 68 / 67 / 85 (InstanceOf / SubClassOf / relational) against 87 / 81 / 93.
       The authors' code reproduces the paper. With every ingredient of its training loop
       and its starting values and test rule replaced by mOWL's it still reaches
       80 / 63 / 89, so the InstanceOf gap is in mOWL's training and is open (see the
       notes of the recipe module ``mowl.benchmarks.recipe_sets.wang2024_eike``).
   * - ELK-Em
     - ``elkem-*``
     - partly
     - GO and ANATOMY close, GALEN below.
   * - EmEL++ (own benchmarks)
     - ``emelpp-*-prediction``
     - partly
     - GO and GALEN (1 seed): close but below. ANATOMY (2 seeds): EmEL++ reproduced;
       the ELEm baseline of that table is not (0.21 / 0.41).
   * - BoxLitE
     - ``boxlite-*``
     - not reproduced
     - Needs the commercial MOSEK solver; with the free ECOS solver the results are
       lower.
   * - Faithfulness benchmark (ELEm, EmEL++, Box\ :sup:`2`\ EL)
     - ``faithfulness-*``
     - partly
     - The paper does not print its hyperparameters; 1 seed.
   * - VolumeMultiBoxEL, OctagonEL, BinaryCodeBox, ALConeModel, E2R, EmEL(var)
     - none
     - no benchmark
     - The reason is recorded on each catalogue card.

The methods that are not in this table (the order, graph, query and region methods added
with the parts) are compared with their publications in the three-way table of
:doc:`index`: published number, mOWL built from parts, and the direct baseline code.

Runs that cannot be made
------------------------

* **FaithEL** on the joint benchmarks (GO, GALEN, ANATOMY, PPI): the canonical model the
  method is defined on would have between 3e8 and 6e9 elements for these ontologies; the
  model refuses to build it.
* **BoxSEL** on Person, Country and Hybrid (``boxsel-sel-*``): the Statistical EL ontologies
  of the paper are not released. :mod:`mowl.datasets.builtin.sel_conditionals` builds one
  from assertions.
* **AConE** on ``query-wn18rr-qa``: the query sets of WN18RR are not downloaded by mOWL;
  give ``data_dir``.
* **HiT and OnT** need the optional language-model packages and a GPU with enough memory
  for the encoder.

Which epoch is kept, and what it did to the joint benchmarks
------------------------------------------------------------

Up to mOWL 2.1.0 an EL model kept the epoch with the lowest loss of the validation axioms.
That loss looks at true axioms only, and for several box models it is lowest before training
has separated true from false axioms (all boxes still overlap). In the first round of the
joint benchmarks 84 of 758 runs kept such an untrained state, among them every BoxSEL run and
most Concept2Box runs, which then ranked at chance level (AUC 0.50).

The default is now ``validation_criterion="separation"``: the epoch is kept at which the
validation axioms score better than corrupted copies of themselves most often
(:func:`mowl.evaluation.validation.corrupted_pair_error`). ``validation_criterion="loss"``
gives the rule of 2.1.0. The affected runs were repeated with the new rule (AUC, one seed):

.. list-table::
   :header-rows: 1
   :widths: 40 20 20

   * - Benchmark
     - BoxSEL
     - Concept2Box
   * - ``common-go-subsumption-prediction``
     - 0.50 → 0.74
     - 0.91
   * - ``common-go-subsumption-inference``
     - 0.50 → 0.76
     - 0.93
   * - ``common-go-existential-prediction``
     - 0.50 → 0.91
     - 0.87
   * - ``common-galen-subsumption-prediction``
     - 0.50 → 0.81
     - 0.92
   * - ``common-galen-subsumption-inference``
     - 0.50 → 0.61
     - 0.52
   * - ``common-galen-existential-prediction``
     - 0.51 → 0.89
     - 0.87
   * - ``common-anatomy-subsumption-prediction``
     - 0.50 → 0.89
     - 0.92
   * - ``common-anatomy-subsumption-inference``
     - 0.50 → 0.89
     - 0.94
   * - ``common-anatomy-existential-prediction``
     - 0.51 → 0.94
     - 0.94

Still open: ``common-galen-subsumption-inference`` stays close to chance for the box models
(BoxSEL 0.61, Concept2Box 0.52, EIKE 0.54, ELBE baseline 0.50). The baseline code with its
own training regimes does not use this rule and its results did not change.

Numbers
-------

Box\ :sup:`2`\ EL, TransBox and the baselines (Hits\@100, combined over normal forms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 15 15 15

   * - Method
     - Benchmark family
     - GALEN
     - GO
     - ANATOMY
   * - Box\ :sup:`2`\ EL
     - ``box2el-*-prediction``
     - 0.31 / 0.35
     - 0.59 / 0.59
     - 0.66 / 0.70
   * - Box\ :sup:`2`\ EL
     - ``box2el-*-inference``
     - 0.24 / 0.24
     - 0.49 / 0.49
     - 0.47 / 0.44
   * - ELEm (baseline code)
     - ``box2el-*-prediction``
     - 0.29 / 0.29
     - 0.44 / 0.43
     - 0.64 / 0.64
   * - TransBox
     - ``transbox-*-prediction``
     - 0.19 / 0.18
     - 0.36 / 0.35
     - 0.63 / 0.62

The Box\ :sup:`2`\ EL rows are runs with the role inclusion and role chain losses of the
paper (``load_role_axioms=True`` in the recipes). Without them, as mOWL trained
Box\ :sup:`2`\ EL before, GALEN gives 0.29 (prediction) and 0.23 (inference) and GO
inference 0.48; the other cells do not change at this precision. With one seed each,
differences of 0.01 are not evidence of an effect.

Under the test design of the TransBox paper, Box\ :sup:`2`\ EL on GO gives 0.51 / 0.52
and ELBE on GO 0.23 / 0.22.

OnT and HiT (Hits\@1, Hits\@10, Hits\@100)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Benchmark
     - OnT
     - HiT
   * - ``ont-galen-prediction``
     - 0.25, 0.50, 0.69 / 0.25, 0.50, 0.69
     - 0.24, 0.47, 0.63 / 0.25, 0.47, 0.62
   * - ``ont-go-prediction``
     - 0.38, 0.68, 0.81 / 0.37, 0.67, 0.81
     - 0.36, 0.63, 0.75 / 0.36, 0.60, 0.73
   * - ``ont-anatomy-prediction``
     - 0.22, 0.57, 0.81 / 0.22, 0.57, 0.82
     - 0.19, 0.54, 0.77 / 0.19, 0.54, 0.78
   * - ``ont-galen-inference``
     - 0.00, 0.05, 0.28 / 0.00, 0.05, 0.28
     - 0.00, 0.05, 0.29 / 0.00, 0.04, 0.26
   * - ``ont-go-inference``
     - 0.00, 0.10, 0.40 / 0.00, 0.10, 0.40
     - 0.04, 0.15, 0.44 / 0.00, 0.01, 0.04
   * - ``ont-anatomy-inference``
     - 0.00, 0.06, 0.40 / 0.00, 0.06, 0.41
     - 0.00, 0.07, 0.42 / 0.00, 0.06, 0.44

CatE (filtered)
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 16 18

   * - Benchmark
     - Mean rank
     - Hits\@10
     - Hits\@100
     - AUC
   * - ``cate-foodon-completion``
     - 2740 / 2764
     - 0.31 / 0.30
     - 0.47 / 0.47
     - 0.90 / 0.90
   * - ``cate-go-completion``
     - 2894 / 2968
     - 0.27 / 0.22
     - 0.58 / 0.58
     - 0.94 / 0.93
   * - ``cate-ore1-membership``
     - 42 / 37
     - 0.63 / 0.51
     - 0.88 / 0.96
     - 0.995 / 0.99

For ORE1 the released training ontology contains no existential restrictions while the
released lattice file contains 3,675, so the published run used a graph that was not
released. The recipe uses the norm :math:`p = 1`, the default of the reference run script;
this value is not stated in the paper and was chosen after comparing one seed of
:math:`p = 1` and :math:`p = 2` on the test set.

DELE (macro mean rank)
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Benchmark
     - ELEm (base, +l)
     - ELBE (base, +l)
     - Box\ :sup:`2`\ EL (base, +l)
   * - ``dele-foodon-subsumption``
     - 4701 / 4659, 5062 / 5015
     - 7020 / 6695, 4908 / 5428
     - 3895 / 3900, 7517 / 7550
   * - ``dele-galen-subsumption``
     - 9268 / 9106, 9043 / 8977
     - 11881 / 11236, 11606 / 11884
     - 11724 / 11724, ~10600 / 11371
   * - ``dele-yeast-ppi``
     - 724 / 599, 355 / 533 (1 seed)
     - 830 / 830, 787 / 985
     - 214 / 215, 202 / 201
   * - ``dele-yeast-function``
     - 7542 / 21198, 11180 / 9603 (1 seed)
     - 9019 / 4229, 17759 / 12920 (1 seed)
     - 1921 / 1988, 2320 / 2129

The released ELEm code of DELE cannot be run as it is (an attribute is never set), so the
exact version behind the ELEm numbers is not known.

RegD (F1)
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Task, ontology (d = 10)
     - RegD boxes
     - RegD balls
     - ELBE
     - ELEm
   * - inference, GALEN
     - 0.18 / 0.26
     - 0.17 / 0.19
     - 0.22 / 0.21
     - 0.17 / 0.17
   * - inference, GO
     - 0.49 / 0.51
     - 0.30 / 0.40
     - 0.69 / 0.42
     - 0.68 / 0.27
   * - inference, ANATOMY
     - 0.41 / 0.63
     - 0.45 / 0.56
     - 0.74 / 0.43
     - 0.70 / 0.39
   * - prediction, GALEN
     - 0.21 / 0.21
     - 0.18 / 0.18
     - 0.21 / 0.24
     - 0.22 / 0.17
   * - prediction, GO
     - 0.55 / 0.44
     - 0.42 / 0.61
     - 0.39 / 0.38
     - 0.62 / 0.54
   * - prediction, ANATOMY
     - 0.26 / 0.45
     - 0.26 / 0.22
     - 0.23 / 0.26
     - 0.25 / 0.26

The prediction numbers are computed as the reference code computes them: its
``compute_nf3_ranks`` reads the positive score from the column that holds the role index
and uses it as a class index. The evaluation regime ``regd-sampled-f1`` reports this value
as ``f1`` and the score of the query's own class as ``f1_pair``.

Other methods
~~~~~~~~~~~~~

* **ELK-Em** (Hits\@100): GO 0.69 / 0.71, ANATOMY 0.49 / 0.50, GALEN 0.31 / 0.35.
* **EmEL++ on its own benchmarks** (Hits\@100): GO 0.22 / 0.24 and GALEN 0.12 / 0.16
  (1 seed); ANATOMY 0.40 / 0.39 (2 seeds; median rank 445 / 381). The ELEm baseline of
  the ANATOMY table is not reproduced: 0.21 / 0.41, median rank about 15,500 / 640.
* **BoxLitE** (MRR with the ECOS solver, family ontology): 0.66 / 0.72 for the first
  variant, down to 0.22 / 0.44 for the fourth.

Implemented, without a published benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods are implemented and tested (loss semantics on hand-built cases and a small
end-to-end run) and can be trained on any ontology, but no number of their publications
can be reproduced. The full reasoning is on each catalogue card.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Method
     - Why there is no benchmark
   * - :class:`mowl.models.VolumeMultiBoxEL` (Aharoni, 2024)
     - The ranking score is not fully specified (number of sampled points, which box,
       aggregation), the test sub-sampling is not described, and the code is not
       available.
   * - :class:`mowl.models.OctagonEL` (Charpenay and Schockaert, 2024)
     - The paper evaluates octagons as relation regions for knowledge-graph link
       prediction; mOWL uses them as concept regions (an adaptation).
   * - :class:`mowl.models.BinaryCodeBox` (Zhang et al., 2022)
     - Every published number comes from a per-graph hyperparameter search whose
       selected values are not published.
   * - :class:`mowl.models.ALConeModel` and the exact al-cone construction (Özçep et
       al., 2020; 2023)
     - The papers are theoretical; learning and evaluation are left to future work. The
       learner is mOWL's own.
   * - :class:`mowl.models.E2R` (Garg et al., 2019)
     - FB15K and WN18 are knowledge-graph benchmarks; the LUBM protocol and the training
       settings are not specified, and no code was found.
   * - :class:`mowl.models.GroundedEL` (original to mOWL)
     - No publication yet; its guarantees are tested on small ontologies
       (``tests/models/test_grounded_model.py``).
   * - :class:`mowl.models.EmELVar` (Mohapatra et al., 2021)
     - With the loss as printed, the total loss is never lower than that of EmEL++, so
       the reported improvement is not explained by the paper; the authors' code was not
       available.

Not run
~~~~~~~

``box2el-yeast-ppi``, ``box2el-human-ppi``, ``dele-human-ppi``, ``dele-human-function``
and ``cate-yeast-ppi`` have recipes or published numbers but no finished run yet.

Published, parts and direct implementation
------------------------------------------

Each method run as the combination of parts users get and as its direct implementation
(:mod:`mowl.baselines`), with the same recipe and seeds (see :ref:`baseline-benchmarks`).
The table shows one metric per benchmark, method and recipe, on all test axioms: the mean
reciprocal rank where it is reported, otherwise the first of Hits\@10, Hits\@100, AUC, F1,
accuracy, success rate and mean rank. Every metric and subset of every run, with the
differences and the number of seeds, is in :download:`three_way.csv`. Both are written by
``scripts/collect_benchmarks.py RESULTS --out report --headline-rst
docs/source/benchmarks/_headline.rst``.

.. include:: _headline.rst

What had to be made explicit
----------------------------

The reproductions above needed the following differences between a paper's text, its code
and mOWL's defaults to become named regimes or recipe arguments. They are the first things
to check when a new method does not reproduce.

* The meaning of the margin (slack or strict): :class:`mowl.training.SlackMargin`,
  :class:`mowl.training.StrictMargin`.
* Which sides of an axiom are corrupted, how many negatives, and whether they are redrawn
  (:class:`mowl.training.PerColumnCorruption`, :class:`mowl.training.FixedNegatives`,
  ``DELECodeTraining(negative_seed=)``).
* What a step is: all axioms or a random sample per normal form
  (:class:`mowl.training.PerNormalFormMiniBatches`).
* How model selection is done (:class:`mowl.training.ValidationRankSelection`,
  :class:`mowl.training.ValidationF1Selection`).
* The test-time score, which is often not the training loss
  (:mod:`mowl.evaluation.el_scoring`), tie handling, and the candidate set.
* Defects of the reference code that a published number depends on. mOWL's default is
  the correct behaviour; the recipe names the option that gives the published behaviour
  (for example ``gci1_radius_term=False`` for ELEm).
