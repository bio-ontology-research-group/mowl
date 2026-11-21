"""Logical query answering benchmarks: the released query datasets of Query2Box and BetaE
(FB15k, FB15k-237, NELL995), for Query2Box, BetaE, ConE and AConE."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import (HE2024ACONE, REN2020BETAE, REN2020Q2B,
                                           ZHANG2021CONE)

_POSITIVE = ("1p", "2p", "3p", "2i", "3i", "pi", "ip", "2u", "up")
_NEGATION = ("2in", "3in", "inp", "pin", "pni")
_METRICS = ("mrr", "hits@1", "hits@3", "hits@10")

_DESC = (
    "{label}: train on the training graph (every entity a class, every edge h ⊑ ∃r.t; "
    "mowl.datasets.builtin.query_sets.QueryAnsweringDataset), then rank, for every released "
    "{split} query, each hard answer (an answer that needs at least one edge of the next "
    "split) among all entities after removing the other answers of the query (filtered "
    "setting). The reciprocal rank and Hits@k are averaged over the hard answers of a "
    "query, then over the queries of a structure; 'average_epfo' is the mean over the nine "
    "positive structures{neg}, 'average' over all structures reported.")
_NOTES = (
    "Closed world with negation as absence (WorldAssumption.CLOSED_STATED): the answers of a "
    "query, and of its negation, are computed on the graph of stated edges, and an entity "
    "that no stated edge makes an answer of q is an answer of ¬q. Benchmarks of the open "
    "world kind would ask for the entities entailed to be non-answers; here a negated query "
    "has many answers. Tied scores share the average rank; the reference code ranks by "
    "argsort and so breaks ties arbitrarily (region scores of Query2Box and ConE tie "
    "rarely). Unions are answered with the disjunctive normal form (best branch). A model "
    "without negation (Query2Box) is evaluated on the structures without negation only. "
    "The recipes train on the released training queries (queries='released'); the model "
    "arguments, not the benchmark, decide this.")


def _benchmark(kg, label, variant):
    betae = variant == "betae"
    names = _POSITIVE + (_NEGATION if betae else ())
    groups = ("average_epfo",) + (("average_negation",) if betae else ()) + ("average",)
    source = ("BetaE (Ren and Leskovec 2020), the queries of BetaE, ConE and AConE: the "
              "nine positive structures, the five with negation (2in, 3in, inp, pin, pni) "
              "and unions; the answers of validation and test queries are capped"
              if betae else
              "Query2Box (Ren et al. 2020): the nine positive structures, without a cap on "
              "the number of answers")
    return Benchmark(
        key=f"query-{kg}-{variant}", name=f"{label} logical queries ({variant.upper()})",
        task=Task.QUERY_ANSWERING, evaluation_regime="query-answering",
        description=_DESC.format(label=f"{label}, {source}", split="test",
                                 neg=" and 'average_negation' over the five with negation"
                                 if betae else ""),
        dataset="mowl.datasets.builtin.query_sets.QueryAnsweringDataset",
        dataset_kwargs={"name": {"fb15k": "FB15k", "fb15k237": "FB15k237",
                                 "nell995": "NELL995"}[kg], "variant": variant},
        protocol="mowl.benchmarks.protocols.bindings.QueryAnsweringProtocol",
        protocol_kwargs={"split": "test"},
        metrics=_METRICS, subsets=names + groups,
        higher_is_better={m: True for m in _METRICS},
        references=((REN2020BETAE, ZHANG2021CONE, HE2024ACONE) if betae
                    else (REN2020Q2B, REN2020BETAE)),
        notes=_NOTES)


BENCHMARKS = [_benchmark(kg, label, variant)
              for variant in ("betae", "q2b")
              for kg, label in (("fb15k", "FB15k"), ("fb15k237", "FB15k-237"),
                                ("nell995", "NELL995"))]

BENCHMARKS.append(Benchmark(
    key="query-wn18rr-qa", name="WN18RR-QA logical queries",
    task=Task.QUERY_ANSWERING, evaluation_regime="query-answering",
    description=_DESC.format(
        label="WN18RR-QA, the query set of Huang et al. (LinE, 2022) in the format of the "
              "BetaE release, used by AConE", split="test",
        neg=" and 'average_negation' over the five with negation"),
    dataset="mowl.datasets.builtin.query_sets.QueryAnsweringDataset",
    dataset_kwargs={"name": "WN18RR-QA", "data_dir": None},
    protocol="mowl.benchmarks.protocols.bindings.QueryAnsweringProtocol",
    protocol_kwargs={"split": "test"},
    metrics=_METRICS,
    subsets=_POSITIVE + _NEGATION + ("average_epfo", "average_negation", "average"),
    higher_is_better={m: True for m in _METRICS},
    references=(HE2024ACONE,),
    notes=_NOTES + " The data are NOT downloaded by mOWL: WN18RR-QA is published on a Google "
    "Drive (linked from the README of https://github.com/RoyaHe/AConE) and has to be "
    "unpacked by hand; pass its directory as dataset_kwargs data_dir (the layout of one "
    "folder of KG_data.zip). The published numbers are recorded but could not be checked "
    "against the data."))
