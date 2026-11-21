"""The EmEL++ release data (Mondal et al., 2021)."""

from mowl.datasets.builtin.mondal2021 import _EMPTY_CHAIN


def test_empty_role_chains_of_the_release_are_recognised():
    """The released ANATOMY ontology has role chains without roles, which no OWL parser
    accepts; the dataset builder drops exactly those lines."""
    assert _EMPTY_CHAIN.search("SubObjectPropertyOf(ObjectPropertyChain( ) <http://x/r>)")
    assert _EMPTY_CHAIN.search("SubObjectPropertyOf(ObjectPropertyChain() <http://x/r>)")
    assert not _EMPTY_CHAIN.search(
        "SubObjectPropertyOf(ObjectPropertyChain(<http://x/r> <http://x/s>) <http://x/t>)")
    assert not _EMPTY_CHAIN.search("SubClassOf(<http://x/A> <http://x/B>)")
