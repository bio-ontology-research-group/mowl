First steps
===========

The fastest way to train and evaluate a method is :func:`mowl.pipeline.pipeline`. It takes
the name of a method and the name of a benchmark, builds the model, trains it, evaluates
it and returns the trained model together with the metrics. It starts the Java virtual
machine itself if it is not running.

Training and evaluating a model
-------------------------------

.. testcode::

   from mowl.pipeline import pipeline

   result = pipeline("elbe", benchmark="helly-tk", epochs=2, seed=0,
                     model_kwargs={"embed_dim": 2})

.. testoutput::
   :options: +ELLIPSIS

   Epoch 1: Train loss: ...

The methods are those of the catalogue and the benchmarks those of the benchmark set; both
can be listed without training anything:

.. testcode::

   from mowl.catalog import list_methods
   from mowl.benchmarks import list_benchmarks, methods_for

   print(len(list_methods(implemented=True)) > 30, len(list_benchmarks()) > 50)
   print("elbe" in [m.key for m in methods_for("helly-tk")])

.. testoutput::

   True True
   True

Reading the result
------------------

The result holds the trained model, the dataset and the record of the run. The metrics are
grouped by subset (for example by normal form, with ``"combined"`` for all of them); a
benchmark with a single subset calls it ``"all"``.

.. testcode::

   print(type(result.model).__name__)
   print(sorted(result.metrics))
   print(0.0 <= result.metric("axiom_satisfaction") <= 1.0)
   print(result.result.method, result.result.epochs, result.result.seed)

.. testoutput::

   ELBE
   ['all']
   True
   elbe 2 0

``result.model`` is an ordinary mOWL model: its embeddings and its PyTorch module are used
as in the rest of this documentation.

.. testcode::

   print(result.model.module.class_embed.weight.shape[1])

.. testoutput::

   2

Saving and loading
------------------

``save_to_directory`` writes ``results.json`` (metrics, settings, seed, timings and software
versions), ``pipeline.json`` (what the pipeline was called with) and ``model.pt`` (the
parameters of the model's PyTorch module). :func:`mowl.pipeline.load_pipeline` reads them
back: it loads the dataset again, builds the model with the arguments of the run and puts
the saved parameters into it. Nothing is trained or evaluated again.

.. testcode::

   import os, tempfile
   import torch as th
   from mowl.pipeline import load_pipeline

   directory = os.path.join(tempfile.mkdtemp(), "elbe_helly")
   result.save_to_directory(directory)
   print(sorted(os.listdir(directory)))

   loaded = load_pipeline(directory)
   print(type(loaded.model).__name__, loaded.metrics == result.metrics)
   print(th.equal(loaded.model.module.class_embed.weight,
                  result.model.module.class_embed.weight))

.. testoutput::

   ['model.pt', 'pipeline.json', 'results.json']
   ELBE True
   True

A run on a :class:`mowl.datasets.Dataset` object of your own is loaded with
``load_pipeline(directory, dataset=...)``, because an object cannot be written to the
directory; a run on ontology files is loaded from the same files.

Using your own ontology
-----------------------

Instead of a benchmark, give a dataset: a :class:`mowl.datasets.Dataset`, the path of an
ontology file, or the paths of the training, validation and testing ontologies. Without
``evaluation`` the model is only trained.

.. testcode::

   def write(path, axioms):
       with open(path, "w") as f:
           f.write("Prefix(:=<http://example.org/>)\nOntology(<http://example.org/" +
                   os.path.basename(path) + ">\n" + "\n".join(axioms) + "\n)\n")
       return path

   folder = tempfile.mkdtemp()
   train = write(os.path.join(folder, "train.owl"), [
       "SubClassOf(:Dog :Mammal)", "SubClassOf(:Cat :Mammal)", "SubClassOf(:Mammal :Animal)",
       "SubClassOf(:Bird :Animal)", "SubClassOf(:Dog ObjectSomeValuesFrom(:eats :Meat))",
       "SubClassOf(:Cat ObjectSomeValuesFrom(:eats :Meat))", "SubClassOf(:Meat :Food)"])

   trained = pipeline("elem", dataset=train, epochs=5, model_kwargs={"embed_dim": 8})
   print(type(trained.model).__name__, trained.metrics)

.. testoutput::
   :options: +ELLIPSIS

   Epoch 1: Train loss: ...
   ELEmbeddings {}

With ``evaluation``, the dataset is evaluated as a benchmark evaluates its own data: the
key names the benchmark whose protocol is used. Here the held-out axioms of the testing
ontology are ranked as in the Box²EL prediction benchmarks.

.. testcode::

   valid = write(os.path.join(folder, "valid.owl"), ["SubClassOf(:Cat :Animal)"])
   test = write(os.path.join(folder, "test.owl"), ["SubClassOf(:Dog :Animal)"])

   evaluated = pipeline(
       "elem", dataset={"ontology": train, "validation": valid, "testing": test},
       evaluation="box2el-go-prediction", epochs=5, model_kwargs={"embed_dim": 8})
   print(0.0 < evaluated.metric("mrr") <= 1.0)

.. testoutput::
   :options: +ELLIPSIS

   Epoch 1: Train loss: ...
   True

Using the trained model for predictions
---------------------------------------

A trained model scores axioms: ``model.score(axiom)`` takes an OWL axiom and returns its
cost under the model, where a lower value means a better fit. Ranking candidate axioms by
this score is how predictions are made. Here the model trained on the small ontology above
is trained for longer and asked which of three classes is the most plausible superclass of
``Dog``.

.. testcode::

   from mowl.owlapi import OWLAPIAdapter

   trained = pipeline("elem", dataset=train, epochs=300, model_kwargs={"embed_dim": 8})
   adapter = OWLAPIAdapter()

   def named(name):
       return adapter.create_class("http://example.org/" + name)

   def cost(sub, sup):
       axiom = adapter.create_subclass_of(named(sub), named(sup))
       return float(trained.model.score(axiom))

   candidates = ["Bird", "Food", "Mammal"]
   print(sorted(candidates, key=lambda sup: cost("Dog", sup))[0])

.. testoutput::
   :options: +ELLIPSIS

   Epoch 1: Train loss: ...
   Mammal

The evaluation regimes of :doc:`/evaluation/index` do the same over all candidates and
report ranking metrics; the class and relation vectors themselves are in
``trained.model.module``.

Training as a publication did
-----------------------------

A recipe is the configuration with which a publication trained a method on a benchmark
(model arguments, training regime, epochs). ``recipe=True`` uses the recipe of the pair,
if there is exactly one; :func:`mowl.benchmarks.list_recipes` lists them.

.. testcode::

   from mowl.benchmarks import list_recipes

   recipe = list_recipes(benchmark="box2el-go-prediction", method="box2el")[0]
   print(recipe.source.key, recipe.epochs > 0)

.. testoutput::
   :options: +ELLIPSIS

   ... True

.. code-block:: bash

   # the same from the command line
   python -m mowl.benchmarks run box2el-go-prediction --method box2el --recipe --device cuda

Where to go next
----------------

- :doc:`/benchmarks/index`: the benchmarks, the published numbers and how ours compare.
- :doc:`/representation/index`: every method is a combination of parts (geometry, axiom
  scores, training regime) that can be exchanged.
- :doc:`/training/index` and :doc:`/evaluation/index`: training and evaluation regimes.
