#!/usr/bin/env python
"""Write the run manifests of the full-scale benchmark from the registry.

One line per run, TAB-separated::

    tag <TAB> benchmark <TAB> method <TAB> epochs <TAB> seed <TAB> extra CLI arguments

and each line is run as::

    python -m mowl.benchmarks run "$BENCHMARK" --method "$METHOD" --epochs "$EPOCHS" \\
        --seed "$SEED" --device cuda --jvm-memory 10g --workdir W --out OUT.json $EXTRA

Families of runs:

* ``reproduction``: every recipe of a publication on a benchmark for which that
  publication's numbers are recorded, with the recipe's epochs, for the implementation
  ``parts`` (what users get) and ``baseline`` (the direct implementation, through
  ``--plugin mowl.baselines.benchmarks --implementation baseline``), for every seed;
* ``common``: the joint benchmarks (``common-*``) for every method that can run on them
  with its default recipe, implementation ``parts``; ``baseline`` too for the methods
  given with ``--common-baseline``.

Tags are ``<family>__<benchmark>__<method>__<recipe>__<implementation>__s<seed>`` and
unique. Manifests are split by expected cost (``fast``, ``medium``, ``slow``: size of the
dataset and kind of method) so that small runs can be packed several to a GPU.

Examples::

    python scripts/benchmark_manifests.py --out-dir manifests             # 1 seed
    python scripts/benchmark_manifests.py --out-dir manifests --seeds 0,1,2
    python scripts/benchmark_manifests.py --out-dir manifests --common-baseline all
    python scripts/benchmark_manifests.py --toy --epochs 2 --out-dir toy  # local CPU test
"""
import argparse
import collections
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASELINE_ARGS = "--plugin mowl.baselines.benchmarks --implementation baseline"
TOY_PLUGIN = "--plugin tests.benchmarks.toy_benchmarks"

#: Default methods also run as baseline on the joint benchmarks: one or two per family.
DEFAULT_COMMON_BASELINE = (
    "elbe", "box2el", "elem", "transbox", "multiboxel", "octagon", "cate", "e2r", "falcon",
    "dfalc", "query2box", "graph_kge", "dl2vec", "onto2vec", "hyperbolic_cones",
    "order_embeddings", "grounded", "canonical_mixture",
)

# ---------------------------------------------------------------------------------------
# Expected cost
# ---------------------------------------------------------------------------------------

#: dataset size class by a substring of the benchmark key (first match wins)
_SIZE = (
    ("toy-", "small"), ("helly", "small"), ("boxlite-family", "small"),
    ("boxsel-sel", "small"), ("dfalc-revision", "small"), ("faithel-family", "small"),
    ("faithfulness-pizza", "small"), ("cate-ore1", "small"),
    ("galen", "medium"), ("wordnet", "medium"), ("yago39k", "medium"),
    ("kg-inductive", "medium"), ("joie", "medium"), ("cate-foodon", "medium"),
    ("dele-foodon", "medium"), ("dele-yeast-function", "medium"),
)

#: methods whose training is expensive whatever the data (language models, Monte Carlo
#: volumes, grounded solvers, exact canonical models)
_SLOW_METHODS = {"hit", "ont", "mcmultibox", "grounded", "canonical_mixture", "faithel"}
_MEDIUM_METHODS = {"query2box", "betae", "cone_qe", "acone", "reshuffle", "boxe",
                   "expressive", "graph_kge", "composed_graph_kge", "multiboxel",
                   "multielbe", "concept2box"}
_ORDER = ("fast", "medium", "slow")


def size_class(benchmark):
    for pattern, size in _SIZE:
        if pattern in benchmark:
            return size
    return "large"


def cost_class(benchmark, method, epochs):
    """``fast``, ``medium`` or ``slow`` (rough; for packing runs, not a promise)."""
    size = size_class(benchmark)
    if method in _SLOW_METHODS:
        kind = "slow"
    elif method in _MEDIUM_METHODS or int(epochs) > 5000:
        kind = "medium"
    else:
        kind = "fast"
    size_cost = {"small": "fast", "medium": "medium", "large": "slow"}[size]
    if size == "small":
        return "fast" if kind != "slow" else "medium"
    return max(size_cost, kind, key=_ORDER.index)


# ---------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------

Run = collections.namedtuple("Run", "family benchmark method recipe implementation seed "
                                    "epochs extra cost")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(text):
    return _SAFE.sub("-", str(text)).strip("-") or "default"


def tag(run):
    return "__".join([run.family, run.benchmark, run.method, _safe(run.recipe),
                      run.implementation, f"s{run.seed}"])


def line(run):
    return "\t".join([tag(run), run.benchmark, run.method, str(run.epochs), str(run.seed),
                      run.extra])


def _recipe_arg(recipe):
    return recipe.source.key + (f":{recipe.name}" if recipe.name else "")


def _baseline():
    """The baseline implementation (registered by importing its module)."""
    import mowl.baselines.benchmarks as baselines
    return baselines.IMPLEMENTATION


def _extra(recipe_arg, implementation, toy, extra=""):
    parts = []
    if toy:
        parts.append(TOY_PLUGIN)
    parts.append(f"--recipe {recipe_arg}")
    if implementation == "baseline":
        parts.append(BASELINE_ARGS)
    if extra:
        parts.append(extra)
    return " ".join(parts)


def reproduction_runs(seeds, epochs=None, methods=None, benchmarks=None, toy=False,
                      extra=""):
    """``(runs, skipped)``: the reproduction runs and ``{(benchmark, method, recipe,
    implementation): reason}`` for those that cannot run."""
    from mowl.benchmarks.recipes import list_recipes
    from mowl.benchmarks.registry import published_results
    from mowl.catalog import get_method
    published = {(p.benchmark, p.method, p.source.key) for p in published_results()}
    baseline = _baseline()
    runs, skipped = [], {}
    for recipe in list_recipes():
        bench = recipe.benchmark
        if toy:
            if not bench.startswith("toy-") or bench.startswith("toy-common-"):
                continue
            original = bench[len("toy-"):]
        else:
            if bench.startswith("toy-") or bench.startswith("common-"):
                continue
            original = bench
        if (original, recipe.method, recipe.source.key) not in published:
            continue
        if methods and recipe.method not in methods:
            continue
        if benchmarks and bench not in benchmarks:
            continue
        rarg = _recipe_arg(recipe)
        try:
            card = get_method(recipe.method)
        except KeyError:
            skipped[(bench, recipe.method, rarg, "parts")] = "not in the catalogue"
            continue
        if not card.implemented:
            skipped[(bench, recipe.method, rarg, "parts")] = "not implemented in mOWL"
            continue
        n = epochs if epochs is not None else recipe.epochs
        for impl in ("parts", "baseline"):
            if impl == "baseline" and not baseline.supports(recipe.method):
                skipped[(bench, recipe.method, rarg, impl)] = \
                    baseline.unsupported_reason(recipe.method)
                continue
            for seed in seeds:
                runs.append(Run("repro", bench, recipe.method, rarg, impl, seed, n,
                                _extra(rarg, impl, toy, extra),
                                cost_class(bench, recipe.method, n)))
    return runs, skipped


def common_benchmarks(toy=False):
    from mowl.benchmarks.registry import list_benchmarks
    prefix = "toy-common-" if toy else "common-"
    return [b.key for b in list_benchmarks() if b.key.startswith(prefix)]


def common_runs(seeds, baseline_methods=(), epochs=None, methods=None, benchmarks=None,
                toy=False, extra=""):
    """``(runs, skipped)`` of the joint benchmarks."""
    from mowl.benchmarks.comparison import runnable_methods
    from mowl.benchmarks.recipes import list_recipes
    baseline = _baseline()
    runs, skipped = [], {}
    for bench in common_benchmarks(toy):
        if benchmarks and bench not in benchmarks:
            continue
        runnable, not_run = runnable_methods(bench)
        for method, reason in not_run.items():
            if not methods or method in methods:
                skipped[(bench, method, "-", "parts")] = reason
        for method in runnable:
            if methods and method not in methods:
                continue
            recipes = list_recipes(bench, method)
            if len(recipes) != 1:
                skipped[(bench, method, "-", "parts")] = \
                    f"{len(recipes)} recipes on the benchmark (expected one default)"
                continue
            recipe = recipes[0]
            rarg = _recipe_arg(recipe)
            n = epochs if epochs is not None else recipe.epochs
            impls = ["parts"]
            if baseline_methods == "all" or method in baseline_methods:
                if baseline.supports(method):
                    impls.append("baseline")
                else:
                    skipped[(bench, method, rarg, "baseline")] = \
                        baseline.unsupported_reason(method)
            for impl in impls:
                for seed in seeds:
                    runs.append(Run("common", bench, method, rarg, impl, seed, n,
                                    _extra(rarg, impl, toy, extra),
                                    cost_class(bench, method, n)))
    return runs, skipped


def check_unique(runs):
    tags = collections.Counter(tag(r) for r in runs)
    dup = [t for t, n in tags.items() if n > 1]
    if dup:
        raise ValueError(f"Duplicate tags: {dup[:5]}")


def write_manifests(runs, out_dir, split=True):
    """Write ``<family>[-<cost>].tsv`` files; return ``{file name: number of lines}``."""
    os.makedirs(out_dir, exist_ok=True)
    groups = collections.defaultdict(list)
    for r in runs:
        groups[f"{r.family}-{r.cost}" if split else r.family].append(r)
    counts = {}
    for name, members in sorted(groups.items()):
        path = os.path.join(out_dir, name + ".tsv")
        with open(path, "w") as f:
            for r in members:
                f.write(line(r) + "\n")
        counts[name + ".tsv"] = len(members)
    return counts


def summary(runs, skipped):
    """Counts per family, cost and implementation, and runs per benchmark (text)."""
    from mowl.benchmarks.report import table
    out = []
    fam = collections.Counter((r.family, r.implementation, r.cost) for r in runs)
    rows = [[f, i, c, n] for (f, i, c), n in sorted(fam.items())]
    out.append(table(["family", "implementation", "cost", "runs"], rows,
                     title=f"{len(runs)} runs"))
    per = collections.defaultdict(collections.Counter)
    for r in runs:
        per[(r.family, r.benchmark)][r.implementation] += 1
        per[(r.family, r.benchmark)]["cost:" + r.cost] += 1
    rows = [[f, b, c["parts"], c["baseline"], c["cost:fast"], c["cost:medium"],
             c["cost:slow"]] for (f, b), c in sorted(per.items())]
    out.append(table(["family", "benchmark", "parts", "baseline", "fast", "medium", "slow"],
                     rows, title="runs per benchmark"))
    reasons = collections.Counter((k[3], v) for k, v in skipped.items())
    rows = [[impl, n, reason[:110]] for (impl, reason), n in reasons.most_common()]
    out.append(table(["implementation", "skipped", "reason"], rows,
                     title=f"{len(skipped)} (benchmark, method, recipe, implementation) "
                           f"not run"))
    return "\n".join(out)


def _parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", default="0", help="comma-separated seeds (e.g. 0,1,2)")
    p.add_argument("--families", default="reproduction,common")
    p.add_argument("--common-baseline", default=",".join(DEFAULT_COMMON_BASELINE),
                   help="methods also run as baseline on the joint benchmarks: "
                        "comma-separated keys, 'all' or 'none'")
    p.add_argument("--methods", default=None, help="only these methods (comma-separated)")
    p.add_argument("--benchmarks", default=None, help="only these benchmarks")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the recipes' epochs (for tests)")
    p.add_argument("--extra", default="", help="appended to every line's extra arguments")
    p.add_argument("--no-split", action="store_true", help="one manifest per family")
    p.add_argument("--toy", action="store_true",
                   help="the toy copies of tests/benchmarks/toy_benchmarks.py (local test)")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.toy:
        import tests.benchmarks.toy_benchmarks  # noqa: F401  (registers the toy benchmarks)
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = set(args.methods.split(",")) if args.methods else None
    benchmarks = set(args.benchmarks.split(",")) if args.benchmarks else None
    cb = args.common_baseline
    common_baseline = "all" if cb == "all" else (() if cb in ("", "none")
                                                 else tuple(cb.split(",")))
    runs, skipped = [], {}
    families = args.families.split(",")
    if "reproduction" in families:
        r, s = reproduction_runs(seeds, args.epochs, methods, benchmarks, args.toy,
                                 args.extra)
        runs += r
        skipped.update(s)
    if "common" in families:
        r, s = common_runs(seeds, common_baseline, args.epochs, methods, benchmarks,
                           args.toy, args.extra)
        runs += r
        skipped.update(s)
    check_unique(runs)
    counts = write_manifests(runs, args.out_dir, split=not args.no_split)
    with open(os.path.join(args.out_dir, "skipped.tsv"), "w") as f:
        for (b, m, rec, impl), reason in sorted(skipped.items()):
            f.write("\t".join([b, m, rec, impl, reason.replace("\t", " ")]) + "\n")
    text = summary(runs, skipped)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text)
    print(text)
    for name, n in counts.items():
        print(f"{os.path.join(args.out_dir, name)}: {n} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
