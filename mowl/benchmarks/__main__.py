"""Command line interface of the mOWL benchmark set.

::

    python -m mowl.benchmarks list [--task TASK] [--target TARGET] [--method KEY]
    python -m mowl.benchmarks show BENCHMARK
    python -m mowl.benchmarks methods BENCHMARK
    python -m mowl.benchmarks published BENCHMARK [--method KEY] [--format FMT]
    python -m mowl.benchmarks run BENCHMARK --method KEY --epochs N --seed S \\
        --device cuda --param embed_dim=200 --param learning_rate=0.001 --out result.json \\
        [--recipe SOURCE[:NAME]] [--implementation NAME --plugin MODULE]
    python -m mowl.benchmarks compare result.json [--method KEY] [--format FMT]
    python -m mowl.benchmarks compare-methods BENCHMARK [--methods a,b] [--epochs N] \\
        [--seeds 0,1,2] [--device cuda] [--out-dir DIR]
    python -m mowl.benchmarks table result1.json result2.json ... [--format FMT]

``--param``, ``--dataset-param`` and ``--protocol-param`` take ``name=value``; the value is
read as JSON when possible (numbers, ``true``/``false``, lists) and as a string
otherwise.
"""

import argparse
import json
import sys


def parse_assignment(text):
    """``"name=value"`` -> ``(name, value)`` with ``value`` parsed as JSON if possible."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    name, raw = text.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        # Python spellings of the JSON constants are accepted too.
        value = {"True": True, "False": False, "None": None}.get(raw.strip(), raw)
    return name.strip(), value


def _build_parser():
    parser = argparse.ArgumentParser(prog="python -m mowl.benchmarks",
                                     description="mOWL benchmark set")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list benchmarks")
    p.add_argument("--task", default=None,
                   help="a benchmark group (axiom_prediction, ...), a completion task "
                        "(subsumption-prediction, link-prediction, ...) or an inference "
                        "task (subsumption, instance-checking, ...)")
    p.add_argument("--target", default=None,
                   help="'deductive inference', 'inductive prediction' or "
                        "'logical status'")
    p.add_argument("--method", default=None,
                   help="only benchmarks whose inference tasks this method declares")

    p = sub.add_parser("show", help="describe one benchmark")
    p.add_argument("benchmark")

    p = sub.add_parser("methods", help="methods that support the tasks of a benchmark")
    p.add_argument("benchmark")
    p.add_argument("--all", action="store_true",
                   help="include methods without an implementation in mOWL")

    p = sub.add_parser("published", help="published results of a benchmark")
    p.add_argument("benchmark")
    p.add_argument("--method", default=None)
    p.add_argument("--format", default="text", choices=("text", "markdown", "rst"))

    p = sub.add_parser("run", help="train a method on a benchmark and evaluate it")
    p.add_argument("benchmark")
    p.add_argument("--method", required=True, help="catalogue key of the method")
    p.add_argument("--epochs", type=int, default=None,
                   help="training epochs (default: the recipe's, otherwise 100)")
    p.add_argument("--recipe", nargs="?", const=True, default=None, metavar="SOURCE",
                   help="train as the publication did (model arguments, training "
                        "regime, epochs); give the publication key if several "
                        "publications have a recipe, or 'key:name' if one publication "
                        "has several recipes (e.g. DELE's model variants)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gpu-memory-fraction", type=float, default=None,
                   help="cap this run at a fraction of the memory of each GPU (for several "
                        "runs on one GPU: a run that searches for the largest batch that "
                        "fits then stops at its share instead of taking all memory)")
    p.add_argument("--param", action="append", default=[], type=parse_assignment,
                   help="model keyword argument name=value (repeatable)")
    p.add_argument("--dataset-param", action="append", default=[], type=parse_assignment)
    p.add_argument("--protocol-param", action="append", default=[], type=parse_assignment)
    p.add_argument("--training-regime", action="append", default=None,
                   help="catalogue key of a training regime to attach to the model, "
                        "optionally with constructor arguments as JSON after a colon "
                        "(key:{\"name\": value}); repeat to combine regimes")
    p.add_argument("--validate-every", type=int, default=None)
    p.add_argument("--no-select-best", action="store_true",
                   help="keep the last epoch instead of the best validation loss")
    p.add_argument("--workdir", default=None, help="directory for checkpoints")
    p.add_argument("--jvm-memory", default="8g")
    p.add_argument("--out", default=None, help="write the result as JSON here")
    p.add_argument("--format", default="text", choices=("text", "markdown", "rst"))
    p.add_argument("--implementation", default="parts",
                   help="which implementation of the method to run: 'parts' (default, "
                        "the combination of parts users get) or one registered by a "
                        "plugin, e.g. 'baseline' (with --plugin mowl.baselines.benchmarks)")
    p.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                   help="import this module first (it may register implementations, "
                        "model families, benchmarks or recipes); repeatable")

    p = sub.add_parser("compare-methods",
                       help="run every method that supports a benchmark's tasks on it")
    p.add_argument("benchmark")
    p.add_argument("--methods", default=None,
                   help="comma-separated catalogue keys (default: all that can be run)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", default="0", help="comma-separated seeds")
    p.add_argument("--device", default="cpu")
    p.add_argument("--param", action="append", default=[], type=parse_assignment,
                   help="model argument name=value for every method, or "
                        "method.name=value for one method (repeatable)")
    p.add_argument("--out-dir", default=None,
                   help="write one JSON per run here; existing files are reused")
    p.add_argument("--jvm-memory", default="8g")
    p.add_argument("--dry-run", action="store_true",
                   help="only list the methods that would be run and those skipped")
    p.add_argument("--format", default="text", choices=("text", "markdown", "rst"))

    p = sub.add_parser("table", help="tabulate saved results of one benchmark by method")
    p.add_argument("results", nargs="+")
    p.add_argument("--subset", default=None)
    p.add_argument("--format", default="text", choices=("text", "markdown", "rst"))

    p = sub.add_parser("compare", help="compare a saved result with published results")
    p.add_argument("result")
    p.add_argument("--method", default=None)
    p.add_argument("--format", default="text", choices=("text", "markdown", "rst"))
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    from mowl.benchmarks import report
    from mowl.benchmarks.registry import get_benchmark, list_benchmarks

    if args.command == "list":
        rows = [[b.key, ", ".join(b.completion_tasks) or ", ".join(b.inference_tasks),
                 b.target.value, b.name]
                for b in list_benchmarks(task=args.task, target=args.target,
                                         method=args.method)]
        print(report.table(["key", "task", "target", "name"], rows), end="")
    elif args.command == "show":
        b = get_benchmark(args.benchmark)
        encodes = f" (encodes: {', '.join(b.encodes)})" if b.encodes else ""
        print(f"{b.key}: {b.name}\ngroup: {b.task.value}\n"
              f"completion tasks: {', '.join(b.completion_tasks) or '-'}{encodes}\n"
              f"inference tasks: {', '.join(b.inference_tasks)}\n"
              f"target: {b.target.value}\n"
              f"evaluation regime: {b.evaluation_regime}\ndataset: {b.dataset} "
              f"{b.dataset_kwargs}\nprotocol: {b.protocol} {b.protocol_kwargs}\n"
              f"subsets: {', '.join(b.subsets)}\nmetrics: {', '.join(b.metrics)}\n\n"
              f"{b.description}\n\nNotes: {b.notes}")
    elif args.command == "methods":
        from mowl.benchmarks.registry import methods_for
        b = get_benchmark(args.benchmark)
        rows = [[m.key, str(m.expressivity) if m.expressivity is not None else "",
                 "yes" if m.implemented else "no", m.name]
                for m in methods_for(b, implemented=not args.all)]
        print(f"Methods that declare the inference tasks of {b.key} "
              f"({', '.join(b.inference_tasks)}):")
        print(report.table(["key", "expressivity", "in mOWL", "name"], rows), end="")
    elif args.command == "published":
        print(report.published_table(args.benchmark, fmt=args.format, method=args.method),
              end="")
    elif args.command == "run":
        from mowl.benchmarks.implementations import load_plugins
        from mowl.benchmarks.runner import run_benchmark
        load_plugins(args.plugin)
        if args.gpu_memory_fraction is not None:
            import torch
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    torch.cuda.set_per_process_memory_fraction(
                        args.gpu_memory_fraction, index)
        result = run_benchmark(
            args.benchmark, method=args.method, model_kwargs=dict(args.param),
            epochs=args.epochs, seed=args.seed, device=args.device,
            training_regime=args.training_regime,
            dataset_kwargs=dict(args.dataset_param) or None,
            protocol_kwargs=dict(args.protocol_param) or None,
            validate_every=args.validate_every, select_best=not args.no_select_best,
            recipe=args.recipe, implementation=args.implementation,
            workdir=args.workdir, jvm_memory=args.jvm_memory)
        if args.out:
            result.save(args.out)
        print(report.result_table(result, fmt=args.format), end="")
    elif args.command == "compare-methods":
        from mowl.benchmarks.comparison import (compare_methods, methods_table,
                                                runnable_methods)
        run, skipped = runnable_methods(args.benchmark)
        methods = args.methods.split(",") if args.methods else run
        if args.dry_run:
            print("would run: " + ", ".join(m for m in methods if m not in skipped))
            for method, reason in skipped.items():
                if not args.methods or method in methods:
                    print(f"skipped {method}: {reason}")
            return 0
        shared, per_method = {}, {}
        for name, value in args.param:
            if "." in name and name.split(".", 1)[0] in methods:
                method, name = name.split(".", 1)
                per_method.setdefault(method, {})[name] = value
            else:
                shared[name] = value
        model_kwargs = {m: {**shared, **per_method.get(m, {})} for m in methods}
        outcome = compare_methods(
            args.benchmark, methods=methods, epochs=args.epochs,
            seeds=[int(s) for s in args.seeds.split(",")], device=args.device,
            model_kwargs=model_kwargs, out_dir=args.out_dir, jvm_memory=args.jvm_memory)
        print(methods_table(outcome.results, fmt=args.format), end="")
        for method, reason in outcome.skipped.items():
            print(f"skipped {method}: {reason}")
        for (method, seed), error in outcome.failed.items():
            print(f"failed {method} (seed {seed}): {error}")
    elif args.command == "table":
        from mowl.benchmarks.comparison import methods_table
        from mowl.benchmarks.runner import BenchmarkResult
        results = [BenchmarkResult.load(path) for path in args.results]
        print(methods_table(results, subset=args.subset, fmt=args.format), end="")
    elif args.command == "compare":
        from mowl.benchmarks.runner import BenchmarkResult, compare
        result = BenchmarkResult.load(args.result)
        rows = compare(result, method=args.method)
        if not rows:
            print(f"No published results for {result.benchmark} / "
                  f"{args.method or result.method}.")
        else:
            print(report.comparison_table(rows, fmt=args.format), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
