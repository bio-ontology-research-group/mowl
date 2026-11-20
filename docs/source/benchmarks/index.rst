Benchmarks
==========

.. contents::
   :local:
   :depth: 2

The mOWL benchmark set collects the benchmarks used in the ontology embedding literature
together with their evaluation protocols and the numbers published for them, so that any
method of the :doc:`method catalogue </methods/index>` can be trained on a benchmark and
compared with the literature under the same protocol.

A **benchmark** (:class:`mowl.benchmarks.Benchmark`) names a dataset, a task and a
**protocol**. The protocol binds an *evaluation regime* of :mod:`mowl.evaluation` to the
benchmark; the regime fixes everything that a sentence like "we evaluate on GALEN" leaves
open: which axioms are queries, which side of the axiom is ranked, the candidate set,
whether known positives are filtered, how ties are broken, and how every metric
(including the ROC AUC, which papers compute in at least three different ways) is
computed. A **published result** (:class:`mowl.benchmarks.PublishedResult`) is one row of
a table of a paper, with its table locator and the settings the paper states. Published
numbers were transcribed from the PDFs and checked against the rendered pages.

Listing benchmarks and published results needs neither the JVM nor any download.

Evaluation protocols
--------------------

Box\ :sup:`2`\ EL subsumption prediction (``box2el-{galen,go,anatomy}-prediction``)
   Data: the 80/10/10 split per normal form of the Box\ :sup:`2`\ EL release
   (:class:`~mowl.datasets.builtin.GALENJackermeier2024Dataset` and its GO and Anatomy
   counterparts). For each test axiom one atomic concept is replaced by *every class of
   the ontology*: the superclass :math:`D` of :math:`C \sqsubseteq D` (:math:`C` itself is
   removed from the candidates), the superclass :math:`E` of
   :math:`C \sqcap D \sqsubseteq E` (:math:`C` and :math:`D` removed), the subclass
   :math:`C` of :math:`C \sqsubseteq \exists r.D`, and the superclass :math:`D` of
   :math:`\exists r.C \sqsubseteq D`. Ranks are raw (no filtering) and ties are broken by
   candidate position. Reported per normal form (subsets ``gci0`` to ``gci3`` = NF1 to
   NF4) and for all ranks combined: Hits@1/10/100, median rank, MRR, mean rank, and the
   AUC of the reference code (trapezoid of TPR over FPR). Disjointness axioms are not
   evaluated. Scores are the centre distances of the paper's Appendix F for
   Box\ :sup:`2`\ EL, ELBE and ELEmbeddings, and the model's own loss for other models;
   the result records which was used. Regime:
   :class:`~mowl.evaluation.normal_form_ranking.NormalFormRankingRegime`.

Approximating deductive reasoning (``box2el-{galen,go,anatomy}-inference``)
   Train on the whole normalised ontology; rank the atomic subsumptions that an
   :math:`\mathcal{EL}^{++}` reasoner infers but that are not asserted (90% test, 10%
   validation, from the release), exactly as NF1 above.
   Data: :class:`~mowl.datasets.builtin.jackermeier2024_reference.Jackermeier2024InferencesDataset`.

ELK-Em (``elkem-{galen,go,anatomy}-subsumption``)
   The same inferred-subsumption data, ranked as the ELK-Em paper does: the superclass
   :math:`D` of every inferred :math:`C \sqsubseteq D` among all classes except
   :math:`\top` and :math:`C`, scored by the margin-free loss :math:`\ell^0`, tied
   candidates at their average rank, Hits@\ :math:`k` over fractional ranks and AUC
   :math:`\mathrm{mean}((n - r)/(n - 1))`. Regime:
   :class:`~mowl.evaluation.normal_form_ranking.ELKEmSubsumptionRankingRegime`; recipe
   and training regime 'elkem-code-training' from
   :mod:`mowl.benchmarks.recipe_sets.singh2026`.

Box\ :sup:`2`\ EL PPI prediction (``box2el-{yeast,human}-ppi``)
   For each test axiom :math:`\{P_1\} \sqsubseteq \exists \mathit{interacts}.\{P_2\}`,
   :math:`P_1` is ranked among all proteins; filtered metrics (``f_`` prefix) move the
   training interactions to the end of the ranking. Data: the ELEmbeddings PPI data of
   the Box\ :sup:`2`\ EL release, which is *not* mOWL's ``PPIYeastDataset``.

DELE (``dele-*``)
   The tail of every test axiom (:math:`A \sqsubseteq B` or
   :math:`\{P\} \sqsubseteq \exists r.\{Q\}`) is ranked among the candidate classes,
   scored by the model's loss, ties averaged; macro metrics average over test axioms and
   micro metrics over head classes; filtered metrics remove the deductive closure of the
   training and test axioms. Regime: :class:`mowl.evaluation.deductive.DELEEvaluation`.
   The DELE paper has yeast data only; the human benchmarks have no published numbers
   (and no public function-prediction data).

CatE (``cate-*``)
   Filtered ranking only: the tail of every test axiom (:math:`C` in :math:`C(a)`,
   :math:`D` in :math:`C \sqsubseteq D`, :math:`P_2` for PPIs) among the named concepts or
   proteins, after removing training axioms (and, for :math:`C(a)`, the deductive closure
   of the training set). Regime: :func:`~mowl.evaluation.filtered_ranking.cate_ranking`.

RegD (``regd-*``)
   1,000 test subsumptions are sampled, 10 random negatives are generated per positive,
   a pair is predicted true iff its score is below the threshold that maximises F1 on
   validation pairs sampled the same way, and F1 is reported. Inference uses the
   reasoner-inferred subsumptions, prediction the axioms
   :math:`A \sqsubseteq \exists r.B` of the 80/10/10 split. Regime:
   :class:`~mowl.evaluation.sampled_f1.SampledF1Regime`.

EmEL++ (``emelpp-*``)
   For every test pair :math:`C \sqsubseteq D` of the 70/20/10 split, all classes but
   :math:`D` are ranked by the distance of their centre from the centre of :math:`D`; the
   rank of :math:`C` is recorded. The validation and test pairs are removed from the
   training ontology (the reference code trains on all of them).

Helly family (``helly-tk``)
   An expressivity benchmark: train on :math:`\mathcal{T}_k`
   (:math:`A_i \equiv \bigsqcap_{j \ne i} X_j`, :math:`A_i \sqcap A_j \sqsubseteq \bot`),
   then decide exactly whether the trained regions are a model, whether some satisfiable
   name got an empty region or one without interior, and how many Helly certificates the
   regions violate, for boxes, balls, prisms, octagons and unions of boxes
   (:class:`~mowl.evaluation.exact_model.ExactModelRegime`). No numbers have been
   published for it.

BoxLitE Family (``boxlite-family-v{1..4}``)
   Filtered head/tail ranking of the test role assertions of the Family subsets F_v1 -
   F_v4 (Section 6 and Appendix G of the BoxLitE paper). BoxLitE is *solved* (a conic
   programme through CVXPY), so the runner ignores the epochs; the recipes pin the
   published best lambdas (Table 9). The published numbers were obtained with the
   commercial solver MOSEK, which is not available to mOWL. With the open-source CVXPY
   solvers the same programme returns other near-optimal embeddings, and the metrics
   depend strongly on which one (ECOS: MRR 0.66 on F_v1 against the published 0.72;
   SCS: 0.31), so the published numbers are not reproducible without MOSEK.

Every benchmark's ``notes`` list how the data and protocol differ from the original
publication, and every regime has a card in the catalogue
(:func:`mowl.catalog.list_regimes`).

Python API
----------

.. testcode::

   from mowl.benchmarks import list_benchmarks, get_benchmark, published_results

   keys = [b.key for b in list_benchmarks(task="deductive_inference")]
   print("box2el-go-inference" in keys)

   for r in published_results(benchmark="box2el-galen-prediction", method="box2el",
                              subset="combined"):
       if r.source.key in ("jackermeier2024", "yang2025"):
           print(r.source.key, r.locator, r.metrics["med"])

.. testoutput::

   True
   jackermeier2024 Table 1 (GALEN) 669.0
   yang2025 Table 7 (GALEN) 1360.0

Running a method on a benchmark trains it with a fixed seed and evaluates it with the
benchmark's protocol. The result records the metrics per subset, the settings, the
evaluation and training regimes, the wall time and the package version and git commit,
and can be saved as JSON. :func:`~mowl.benchmarks.runner.compare` lists, for every
published number, ours, the difference, the source and any setting in which the runs
differ (including the training regime: published numbers are only comparable under the
same training and evaluation regime).

.. testcode::

   from mowl.benchmarks.runner import run_benchmark, compare

   result = run_benchmark("helly-tk", method="elbe", model_kwargs={"embed_dim": 2},
                          epochs=2, seed=0)
   print(sorted(result.metrics["all"]))
   print(compare(result))

.. testoutput::
   :options: +ELLIPSIS

   ...['axiom_satisfaction', 'empty_fraction', 'exact_model', 'exact_model_nonempty', 'helly_violation_rate', 'is_model', 'no_interior_fraction']
   []

Models are built through a *family handler*: subclasses of
:class:`~mowl.base_models.EmbeddingELModel` are supported; graph, syntactic and ALC models
raise :class:`~mowl.benchmarks.runner.UnsupportedModelError` until a handler is
registered with :func:`~mowl.benchmarks.runner.register_family`. Any model can also be
passed as ``model_factory(dataset)``.

Command line
------------

The runs on the large ontologies are meant for a GPU cluster; the command line interface
makes them scriptable:

.. code-block:: bash

   python -m mowl.benchmarks list
   python -m mowl.benchmarks show box2el-galen-prediction
   python -m mowl.benchmarks published box2el-galen-prediction --format markdown
   python -m mowl.benchmarks run box2el-galen-prediction --method box2el \
       --epochs 1000 --seed 0 --device cuda \
       --param embed_dim=200 --param learning_rate=0.005 --out galen_box2el_s0.json
   python -m mowl.benchmarks compare galen_box2el_s0.json --format markdown

``--param``, ``--dataset-param`` and ``--protocol-param`` take ``name=value`` (values are
read as JSON when possible). ``--training-regime`` attaches a catalogued training regime;
it can be repeated to combine regimes, and constructor arguments follow the key as JSON
(``--training-regime 'per-normal-form-minibatch:{"batch_size": 512}'``). ``--recipe``
trains the way the publication did (see below).
The papers average several seeds (5 for Box\ :sup:`2`\ EL, 10 for TransBox); run one job
per seed and average the JSON results.

Reproducing published numbers: recipes
--------------------------------------

A published number depends on three things that the paper's method section does not fix:
how the model was trained, how it was tested, and the hyperparameters. mOWL makes each of
them explicit.

* The *evaluation regime* is part of the benchmark. Two papers that test differently on
  the same data get different benchmarks. For example, the TransBox paper re-implements
  the Box\ :sup:`2`\ EL ranking with a different score for
  :math:`C \sqcap D \sqsubseteq E` (every candidate scores 0 when the boxes of
  :math:`C` and :math:`D` are disjoint); its numbers are filed under
  ``transbox-*-prediction``
  (:class:`~mowl.evaluation.normal_form_ranking.TransBoxNormalFormRankingRegime`), not
  under ``box2el-*-prediction``.
* The *training regime* names the training design of the code that produced the numbers:
  :class:`~mowl.training.Box2ELCodeTraining` and
  :class:`~mowl.training.TransBoxCodeTraining`. They train on a random sample of axioms
  per normal form and step, corrupt both sides of :math:`C \sqsubseteq \exists r.D`,
  use a *slack* margin (:class:`~mowl.training.SlackMargin`) and select the model by the
  validation rank of :math:`C \sqsubseteq D`. With a slack margin an inclusion may be
  violated by up to the margin, so the trained embedding is in general not a model of
  the ontology; the design targets ranking by centre distance.
* A :class:`~mowl.benchmarks.Recipe` adds the hyperparameters of the publication.

.. testcode::

   from mowl.benchmarks import get_recipe, list_recipes

   recipe = get_recipe("box2el-go-prediction", "box2el")
   print(recipe.training_regime, recipe.epochs)
   print(sorted(recipe.regime_kwargs.items()))
   print(len(list_recipes(source="jackermeier2024")))

.. testoutput::

   box2el-code-training 10000
   [('margin', 0.15), ('num_negatives', 5)]
   30

``run_benchmark(benchmark, method=..., recipe=True)`` and ``--recipe`` on the command line
apply a recipe; arguments given explicitly take precedence.

.. code-block:: bash

   python -m mowl.benchmarks run box2el-go-prediction --method box2el --recipe \
       --seed 0 --device cuda --out go_box2el_s0.json

Published results
-----------------

The tables below are generated from the registry
(:func:`mowl.benchmarks.report.published_to_rst`).

.. include:: _published.rst
