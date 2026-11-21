"""mOWL behaves as recorded before the Java libraries were last upgraded.

See ``tests/owlapi/compat/README.md``. Each case runs one behaviour probe on one fixture
ontology and compares the result with the golden file.
"""

import faulthandler
import json
import os

import pytest

from tests.owlapi.compat import probes


#: Seconds a probe that uses the KM reasoner may take before the run is aborted.
KM_TIME_LIMIT = 600


def _normalise(value):
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


def test_every_case_has_a_golden_file():
    missing = [f"{o}.{p}" for o, p in probes.cases()
               if not os.path.exists(probes.golden_path(o, p))]
    assert not missing, f"no golden file for {missing}; see compat/README.md"


def test_no_orphan_golden_files():
    expected = {os.path.basename(probes.golden_path(o, p))
                for o, p in probes.cases() + sorted(probes.UNSTABLE)}
    present = {f for f in os.listdir(probes.GOLDEN) if f.endswith(".json")}
    assert present <= expected, sorted(present - expected)


@pytest.mark.parametrize("ontology,probe", probes.cases(),
                         ids=[f"{o}.{p}" for o, p in probes.cases()])
def test_behaviour_matches_golden(ontology, probe):
    path = probes.golden_path(ontology, probe)
    if not os.path.exists(path):
        pytest.fail(f"no golden file {os.path.basename(path)}")
    with open(path) as f:
        golden = json.load(f)
    if probe in probes.USES_KM:
        # The KM backend waits for its helper process without a time limit. If it does
        # not answer, stop the run with a traceback instead of hanging forever.
        faulthandler.dump_traceback_later(KM_TIME_LIMIT, exit=True)
    try:
        current = _normalise(probes.PROBES[probe](probes.ONTOLOGIES[ontology]))
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert current == golden["value"], (
        f"{ontology}.{probe} differs from the behaviour recorded with OWL API "
        f"{golden['owlapi']}")


def test_invariants_recorded_in_the_golden_files_hold():
    """The recorded values themselves say the reference behaviour was sound: saving and
    reloading keeps the axioms, and normalisation keeps the classification."""
    for ontology in probes.ONTOLOGIES:
        with open(probes.golden_path(ontology, "round_trip")) as f:
            for fmt, kept in json.load(f)["value"].items():
                assert kept["axioms"] and kept["entities"], (ontology, fmt)
        path = probes.golden_path(ontology, "normalisation_preserves_classification")
        if os.path.exists(path):
            with open(path) as f:
                assert json.load(f)["value"]["preserved"], ontology
