import sys
sys.path.append("../..")

from mowl.datasets.base import PathDataset

from mowl.graph.graph import GraphGenModel
import mowl.graph.dl2vec.generate_graph as gen
from mowl.graph.edge import Edge
from org.mowl.Parsers import DL2VecParser


if __name__ == "__main__":

    rel_dict = {"subclassof": "subclassof",
                "<http://purl.obolibrary.org/obo/bfo_0000050>": "part_of",
                "<http://purl.obolibrary.org/obo/bfo_0000051>": "has_part",
                "<http://purl.obolibrary.org/obo/bfo_0000066>": "occurs_in",
                "": "equivalentTo",
                "<http://purl.obolibrary.org/obo/ro_0002211>": "regulates",
                "<http://purl.obolibrary.org/obo/ro_0002212>": "negatively_regulates",
                "<http://purl.obolibrary.org/obo/ro_0002213>": "positively_regulates",
                "disjointwith": "disjointwith",
                "<http://purl.obolibrary.org/obo/ro_0002092>": "happens_during",
                "<http://purl.obolibrary.org/obo/go_0005248>": "GO:0005248",
                "<http://purl.obolibrary.org/obo/ro_0002093>": "ends_during",
                "<http://purl.obolibrary.org/obo/go_0007303>": "GO:0007303",
}

    dataset = PathDataset("data/go.owl", "", "")

    edgesOld = {(e.src(), rel_dict[e.rel().lower()], e.dst()) for e in gen.parseOWL(dataset.ontology)}
    edgesNew = DL2VecParser(dataset.ontology, False, "").parse()
    edgesNew = {(str(e.src()), str(e.rel()).lower(), str(e.dst())) for e in edgesNew}

    diff_edges1 = {(s,r,d) for (s,r,d) in edgesOld-edgesNew if not r in ["equivalentTo", "part_of", "regulates", "positively_regulates", "negatively_regulates", "occurs_in", "happens_during", "ends_during", "has_part", "disjointwith"]}
    diff_edges2 = {(s,r,d) for (s,r,d) in edgesNew-edgesOld}


    print(f"Lens: {list(diff_edges1)[:10]} {len(diff_edges1)}")
    print(f"Lens: {list(diff_edges2)[:10]} {len(diff_edges2)}")


    found = [(s, r,d ) for (s,r,d) in edgesNew if (s == "GO:0002278")]

    print("FOUND: ", found)
