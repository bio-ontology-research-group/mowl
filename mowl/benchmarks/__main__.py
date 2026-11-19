"""Command line interface of the mOWL benchmark set.

::

    python -m mowl.benchmarks list [--task TASK]
    python -m mowl.benchmarks show BENCHMARK
    python -m mowl.benchmarks published BENCHMARK [--method KEY] [--format FMT]
    python -m mowl.benchmarks run BENCHMARK --method KEY --epochs N --seed S \\
        --device cuda --param embed_dim=200 --param learning_rate=0.001 --out result.json
    python -m mowl.benchmarks compare result.json [--method KEY] [--format FMT]

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
    p.add_argument("--task", default=None)

    p = sub.add_parser("show", help="describe one benchmark")
    p.add_argument("benchmark")

    p = sub.add_parser("published", help="published results of a benchmark")
    p.add_argument("benchmark")
    p.add_argument("--method", default=None)
    p.add_argument("--format", default="text", choices=("text", "markdown", "rst"))

    p = sub.add_parser("run", help="train a method on a benchmark and evaluate it")
    p.add_argument("benchmark")
    p.add_argument("--method", required=True, help="catalogue key of the method")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--param", action="append", default=[], type=parse_assignment,
                   help="model keyword argument name=value (repeatable)")
    p.add_argument("--dataset-param", action="append", default=[], type=parse_assignment)
    p.add_argument("--protocol-param", action="append", default=[], type=parse_assignment)
    p.add_argument("--training-regime", default=None,
                   help="catalogue key of a training regime to attach to the model")
    p.add_argument("--validate-every", type=int, default=None)
    p.add_argument("--no-select-best", action="store_true",
                   help="keep the last epoch instead of the best validation loss")
    p.add_argument("--workdir", default=None, help="directory for checkpoints")
    p.add_argument("--jvm-memory", default="8g")
    p.add_argument("--out", default=None, help="write the result as JSON here")
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
        rows = [[b.key, b.task.value, b.name] for b in list_benchmarks(task=args.task)]
        print(report.table(["key", "task", "name"], rows), end="")
    elif args.command == "show":
        b = get_benchmark(args.benchmark)
        print(f"{b.key}: {b.name}\ntask: {b.task.value}\ndataset: {b.dataset} "
              f"{b.dataset_kwargs}\nprotocol: {b.protocol} {b.protocol_kwargs}\n"
              f"subsets: {', '.join(b.subsets)}\nmetrics: {', '.join(b.metrics)}\n\n"
              f"{b.description}\n\nNotes: {b.notes}")
    elif args.command == "published":
        print(report.published_table(args.benchmark, fmt=args.format, method=args.method),
              end="")
    elif args.command == "run":
        from mowl.benchmarks.runner import run_benchmark
        result = run_benchmark(
            args.benchmark, method=args.method, model_kwargs=dict(args.param),
            epochs=args.epochs, seed=args.seed, device=args.device,
            training_regime=args.training_regime,
            dataset_kwargs=dict(args.dataset_param) or None,
            protocol_kwargs=dict(args.protocol_param) or None,
            validate_every=args.validate_every, select_best=not args.no_select_best,
            workdir=args.workdir, jvm_memory=args.jvm_memory)
        if args.out:
            result.save(args.out)
        print(report.result_table(result, fmt=args.format), end="")
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
