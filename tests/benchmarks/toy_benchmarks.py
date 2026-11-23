"""Toy copies of registered benchmarks on the tiny ontologies of :mod:`tests.benchmarks.tiny`,
for running the benchmark manifests end to end on a CPU in seconds.

Imported as a plugin of the command line (``--plugin tests.benchmarks.toy_benchmarks``), it
registers, for every benchmark in :data:`TOYS`, a benchmark ``toy-<key>`` with the same
evaluation protocol on the tiny data, and a copy of every recipe of the original benchmark
for the toy one (same model arguments and training regime). The manifest generator
(``scripts/benchmark_manifests.py --toy``) then writes the command lines the cluster would run,
with the toy keys.
"""
from dataclasses import replace

from mowl.benchmarks.recipes import list_recipes, register_recipe
from mowl.benchmarks.registry import get_benchmark, register_benchmark

#: original benchmark -> protocol arguments of the toy copy that differ.
TOYS = {
    # the joint benchmarks: one ranking for every family of methods
    "common-galen-subsumption-prediction": {},
    "common-galen-existential-prediction": {},
    # a reproduction benchmark with a recipe that has a training regime
    "box2el-galen-prediction": {},
}

#: the tiny dataset has no individuals
DATASET = "tests.benchmarks.tiny.tiny_dataset"


def toy_key(key):
    return "toy-" + key


def register():
    out = []
    for key, protocol in TOYS.items():
        original = get_benchmark(key)
        toy = replace(original, key=toy_key(key), name="toy: " + original.name,
                      dataset=DATASET, dataset_kwargs={},
                      protocol_kwargs={**original.protocol_kwargs, **protocol})
        out.append(register_benchmark(toy))
        for recipe in list_recipes(benchmark=key):
            register_recipe(replace(recipe, benchmark=toy.key))
    return out


BENCHMARKS = register()
