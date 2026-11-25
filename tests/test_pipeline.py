"""mowl.pipeline: one call to build, train and evaluate a method."""
import os

import pytest

from mowl.pipeline import PipelineResult, load_pipeline, pipeline
from tests.benchmarks.tiny import scratch_dir, tiny_dataset, write_tiny


def test_benchmark_run_returns_model_metrics_and_saves():
    out = pipeline("elbe", benchmark="helly-tk", epochs=2, model_kwargs={"embed_dim": 2})
    assert isinstance(out, PipelineResult)
    assert type(out.model).__name__ == "ELBE"
    assert out.result.benchmark == "helly-tk" and out.result.method == "elbe"
    assert 0.0 <= out.metric("axiom_satisfaction") <= 1.0
    assert out.metric("axiom_satisfaction") == out.metrics["all"]["axiom_satisfaction"]

    written = out.save_to_directory(os.path.join(scratch_dir(), "run"))
    from mowl.benchmarks.runner import BenchmarkResult
    assert BenchmarkResult.load(written["results"]).metrics == out.metrics
    import torch as th
    state = th.load(written["model"])
    assert set(state) == set(out.model.module.state_dict())


def test_own_ontology_file_is_trained_without_evaluation():
    train, _, _ = write_tiny()
    out = pipeline("elem", dataset=train, epochs=3, model_kwargs={"embed_dim": 4})
    assert out.metrics == {}
    assert out.dataset.ontology.getAxiomCount() > 0
    assert out.model.module.class_embed.weight.shape[1] == 4
    with pytest.raises(ValueError, match="subset"):
        out.metric("mrr")


def test_own_dataset_is_evaluated_like_a_benchmark():
    train, valid, test = write_tiny()
    paths = {"ontology": train, "validation": valid, "testing": test}
    out = pipeline("elem", dataset=paths, evaluation="box2el-go-prediction", epochs=3,
                   model_kwargs={"embed_dim": 4})
    assert out.result.benchmark == "box2el-go-prediction"
    assert 0.0 < out.metric("mrr") <= 1.0

    same = pipeline("elem", dataset=tiny_dataset(), evaluation="box2el-go-prediction",
                    epochs=3, model_kwargs={"embed_dim": 4})
    assert same.metrics == out.metrics, "a Dataset and the paths of its files are the same run"


def test_arguments_that_do_not_go_together_are_refused():
    with pytest.raises(ValueError, match="either a benchmark or a dataset"):
        pipeline("elem")
    with pytest.raises(ValueError, match="either a benchmark or a dataset"):
        pipeline("elem", benchmark="helly-tk", dataset="x.owl")
    with pytest.raises(ValueError, match="its own evaluation"):
        pipeline("elem", benchmark="helly-tk", evaluation="box2el-go-prediction")
    with pytest.raises(ValueError, match="method key or a model_factory"):
        pipeline(benchmark="helly-tk")
    with pytest.raises(ValueError, match="keys 'ontology'"):
        pipeline("elem", dataset={"train": "x.owl"})


def _same_parameters(a, b):
    import torch as th
    sa, sb = a.module.state_dict(), b.module.state_dict()
    return set(sa) == set(sb) and all(th.equal(sa[k], sb[k]) for k in sa)


def test_a_saved_benchmark_run_is_loaded_with_its_parameters():
    out = pipeline("elbe", benchmark="helly-tk", epochs=2, model_kwargs={"embed_dim": 2})
    directory = os.path.join(scratch_dir(), "run")
    out.save_to_directory(directory)
    back = load_pipeline(directory)
    assert back.model is not out.model and type(back.model) is type(out.model)
    assert _same_parameters(back.model, out.model)
    assert back.metrics == out.metrics and back.call == out.call


def test_a_saved_run_on_ontology_files_is_loaded_from_the_same_files():
    train, valid, test = write_tiny()
    paths = {"ontology": train, "validation": valid, "testing": test}
    out = pipeline("elem", dataset=paths, evaluation="box2el-go-prediction", epochs=3,
                   model_kwargs={"embed_dim": 4})
    directory = os.path.join(scratch_dir(), "run")
    out.save_to_directory(directory)
    back = load_pipeline(directory)
    assert _same_parameters(back.model, out.model)
    assert back.model.dataset.class_to_id.keys() == out.model.dataset.class_to_id.keys()

    only_trained = pipeline("elem", dataset=train, epochs=2, model_kwargs={"embed_dim": 4})
    only_trained.save_to_directory(directory + "2")
    assert _same_parameters(load_pipeline(directory + "2").model, only_trained.model)


def test_a_run_on_a_dataset_object_needs_the_dataset_to_load():
    out = pipeline("elem", dataset=tiny_dataset(), epochs=2, model_kwargs={"embed_dim": 4})
    directory = os.path.join(scratch_dir(), "run")
    out.save_to_directory(directory)
    with pytest.raises(ValueError, match="pass it as dataset="):
        load_pipeline(directory)
    back = load_pipeline(directory, dataset=tiny_dataset())
    assert _same_parameters(back.model, out.model)
