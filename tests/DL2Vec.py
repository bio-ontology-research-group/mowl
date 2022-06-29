import sys
sys.path.append("../../")
import mowl
mowl.init_jvm("2g")

from mowl.datasets.base import PathDataset

from mowl.graph.graph import GraphGenModel
import mowl.graph.dl2vec.generate_graph as gen
from mowl.graph.edge import Edge
from org.mowl.Parsers import DL2VecParser
import pickle as pkl

if __name__ == "__main__":

    rel_dict = {"subclassof": "subclassof",
                "<http://purl.obolibrary.org/obo/bfo_0000050>": "<http://purl.obolibrary.org/obo/bfo_0000050>",
                "<http://purl.obolibrary.org/obo/bfo_0000051>": "<http://purl.obolibrary.org/obo/bfo_0000051>",
                "<http://purl.obolibrary.org/obo/bfo_0000066>": "<http://purl.obolibrary.org/obo/bfo_0000066>",
                "": "subclassof",
                "<http://purl.obolibrary.org/obo/ro_0002211>": "<http://purl.obolibrary.org/obo/ro_0002211>",
                "<http://purl.obolibrary.org/obo/ro_0002212>": "<http://purl.obolibrary.org/obo/ro_0002212>",
                "<http://purl.obolibrary.org/obo/ro_0002213>": "<http://purl.obolibrary.org/obo/ro_0002213>",
                "disjointwith": "disjointwith",
                "<http://purl.obolibrary.org/obo/ro_0002092>": "<http://purl.obolibrary.org/obo/ro_0002092>",
                "<http://purl.obolibrary.org/obo/go_0005248>": "GO:0005248",
                "<http://purl.obolibrary.org/obo/ro_0002093>": "<http://purl.obolibrary.org/obo/ro_0002093>",
                "<http://purl.obolibrary.org/obo/go_0007303>": "GO:0007303",
                }

    dataset = PathDataset("data/goslim_yeast.owl", None, None)

    edgesOld = {(e.src(), str(rel_dict[e.rel().lower()]), e.dst()) for e in gen.parseOWL(dataset.ontology) if not str(e.rel()) in ["a"]}
    edgesNew = DL2VecParser(dataset.ontology, False).parse()
    edgesNew = {(str(e.src()), str(e.rel()).lower(), str(e.dst())) for e in edgesNew}

    print("Length old, new: ", len(edgesOld), len(edgesNew))
    
    # edges_old_file = open("data/edges_old.pkl", "wb")
    # edges_new_file = open("data/edges_new.pkl", "wb")
    # pkl.dump(edgesOld, edges_old_file)
    # pkl.dump(edgesNew, edges_new_file)

    diff_edges1 = {(s,r,d) for (s,r,d) in edgesOld-edgesNew} # if not r in  ["equivalentTo"] 
    diff_edges2 = {(s,r,d) for (s,r,d) in edgesNew-edgesOld}


    print(f"Lens: {list(diff_edges1)[:10]} {len(diff_edges1)}")
    print(f"Lens: {list(diff_edges2)[:10]} {len(diff_edges2)}")

    
    print(len(edgesOld))
    print(len(edgesNew))

    ex1 = [(s,r,d) for (s,r,d) in edgesOld if s == "GO:0015171" and d== "GO:0003333"]
    ex2 = [(s,r,d) for (s,r,d) in edgesNew if s == "GO:0015171" and d== "GO:0003333"]


