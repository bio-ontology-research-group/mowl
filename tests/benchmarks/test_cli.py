"""Smoke tests of ``python -m mowl.benchmarks``."""

import json
import os

import mowl  # noqa: F401
from mowl.benchmarks.__main__ import main, parse_assignment
from tests.benchmarks.tiny import scratch_dir


def test_parse_assignment():
    assert parse_assignment("embed_dim=200") == ("embed_dim", 200)
    assert parse_assignment("lr=0.001") == ("lr", 0.001)
    assert parse_assignment("flag=true") == ("flag", True)
    assert parse_assignment("name=GALEN") == ("name", "GALEN")


def test_list_show_published(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "box2el-galen-prediction" in out and "helly-tk" in out
    main(["list", "--task", "exact_model"])
    assert "box2el" not in capsys.readouterr().out
    main(["show", "box2el-go-inference"])
    out = capsys.readouterr().out
    assert "Jackermeier2024InferencesDataset" in out
    assert "inference tasks: subsumption" in out and "target: deductive inference" in out
    main(["list", "--task", "link-prediction", "--target", "inductive prediction"])
    out = capsys.readouterr().out
    assert "box2el-yeast-ppi" in out and "box2el-go-inference" not in out
    main(["list", "--method", "bcbox"])
    out = capsys.readouterr().out
    assert "box2el-go-inference" in out and "box2el-yeast-ppi" not in out
    main(["methods", "box2el-go-inference"])
    out = capsys.readouterr().out
    assert "bcbox" in out and "gda_projection" not in out
    for fmt in ("text", "markdown", "rst"):
        main(["published", "box2el-galen-prediction", "--method", "box2el", "--format", fmt])
        assert "669" in capsys.readouterr().out


def test_run_and_compare(capsys):
    d = scratch_dir()
    out = os.path.join(d, "result.json")
    main(["run", "helly-tk", "--method", "elbe", "--epochs", "2", "--seed", "1",
          "--param", "embed_dim=2", "--dataset-param", f"directory={d}", "--out", out])
    assert "axiom_satisfaction" in capsys.readouterr().out
    data = json.load(open(out))
    assert data["benchmark"] == "helly-tk" and data["seed"] == 1
    main(["compare", out])
    assert "No published results" in capsys.readouterr().out

    # A hand-made result on a benchmark with published numbers.
    data.update(benchmark="box2el-galen-prediction", method="box2el",
                metrics={"combined": {"mr": 5000.0, "med": 700.0}})
    path = os.path.join(d, "fake.json")
    json.dump(data, open(path, "w"))
    main(["compare", path, "--format", "markdown"])
    text = capsys.readouterr().out
    assert "4375" in text and "jackermeier2024" in text
