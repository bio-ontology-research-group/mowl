"""The mOWL benchmark set.

Benchmarks (dataset + task + evaluation protocol) and the results published for them in
the literature, so that any method of :mod:`mowl.catalog` can be run on a benchmark and
compared against published numbers.

.. testcode::

   from mowl.benchmarks import list_benchmarks, published_results

   for b in list_benchmarks(task="deductive_inference"):
       print(b.key, b.name)

   for r in published_results(benchmark="box2el-go-inference"):
       print(r.method, r.metrics, r.source.key, r.locator)

.. testoutput::
   :options: +ELLIPSIS

   ...box2el-go-inference...box2el {...

Listing benchmarks and published results does not need the JVM.
"""

from mowl.benchmarks.spec import Task, Benchmark, PublishedResult
from mowl.benchmarks.registry import (get_benchmark, list_benchmarks, published_results,
                                      supports, missing_support, methods_for,
                                      benchmarks_for)
from mowl.benchmarks.recipes import Recipe, get_recipe, list_recipes

__all__ = ["Task", "Benchmark", "PublishedResult",
           "get_benchmark", "list_benchmarks", "published_results",
           "supports", "missing_support", "methods_for", "benchmarks_for",
           "Recipe", "get_recipe", "list_recipes"]
