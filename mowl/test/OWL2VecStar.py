import sys
sys.path.append("../..")

from mowl.datasets.base import PathDataset

from mowl.graph.graph import GraphGenModel
from mowl.graph.owl2vec_star.model import OWL2VecParser
from mowl.graph.edge import Edge
from org.mowl.Parsers import OWL2VecStarParser
import pickle as pkl

from java.lang import String
from java.util import HashSet


def transformString(string):
    
    rel_dict = {"http://www.w3.org/2000/01/rdf-schema#subclassof": "subclassof",
                "http://www.semanticweb.org/owl2vec#superclassof": "superclassof",
                "http://purl.obolibrary.org/obo/bfo_0000050": "http://purl.obolibrary.org/obo/bfo_0000050",
                "http://purl.obolibrary.org/obo/bfo_0000051": "http://purl.obolibrary.org/obo/bfo_0000051",
                "http://purl.obolibrary.org/obo/bfo_0000066": "http://purl.obolibrary.org/obo/bfo_0000066",
                "http://purl.obolibrary.org/obo/ro_0002211": "http://purl.obolibrary.org/obo/ro_0002211",
                "http://purl.obolibrary.org/obo/ro_0002212": "http://purl.obolibrary.org/obo/ro_0002212",
                "http://purl.obolibrary.org/obo/ro_0002213": "http://purl.obolibrary.org/obo/ro_0002213",
                "http://purl.obolibrary.org/obo/ro_0002092": "http://purl.obolibrary.org/obo/ro_0002092",
                "http://purl.obolibrary.org/obo/ro_0002093": "http://purl.obolibrary.org/obo/ro_0002093",
                }


    if string in rel_dict:
        string = rel_dict[string]

    return string


if __name__ == "__main__":

    dataset = PathDataset("data/goslim_yeast.owl", None, None)

    bd = False
    ot = False
    il = False

    
    parserOld = OWL2VecParser(dataset, bidirectional_taxonomy = bd, only_taxonomy = ot, include_literals = il)
    edgesOld = {(e.src(), transformString(e.rel().lower()), e.dst()) for e in parserOld.parseOWL()}

    edgesNew = OWL2VecStarParser(dataset.ontology, bd, ot, il, HashSet(), HashSet(), HashSet(), String("10240")).parse()
    edgesNew = {(str(e.src()), str(e.rel()).lower(), str(e.dst())) for e in edgesNew}


#    print("Length old, new: ", len(edgesOld), len(edgesNew))
    
#    edges_old_file = open("data/edges_old.pkl", "wb")
#    edges_new_file = open("data/edges_new.pkl", "wb")
#    pkl.dump(edgesOld, edges_old_file)
#    pkl.dump(edgesNew, edges_new_file)

    diff_edges1 = {(s,r,d) for (s,r,d) in edgesOld-edgesNew} # if not r in  ["equivalentTo"] 
    diff_edges2 = {(s,r,d) for (s,r,d) in edgesNew-edgesOld}


    testListOld = [(s,r,d) for (s,r,d) in edgesOld if r == "http://www.geneontology.org/formats/oboinowl#hasdbxref"]

    testListNew = [(s,r,d) for (s,r,d) in edgesNew if r == "http://www.geneontology.org/formats/oboinowl#hasdbxref"]

    print(f"Diff old-new:\n{list(diff_edges1)}")
    print(f"Diff new-old:\n{list(diff_edges2)}")
    print(f"Lengths: {len(diff_edges1)}, {len(diff_edges2)}")

    print("Num edges Old: ", len(edgesOld))
    print("Num edges New: ", len(edgesNew))

    ex1 = [(s,r,d) for (s,r,d) in edgesOld if s == "GO:0015171" and d== "GO:0003333"]
    ex2 = [(s,r,d) for (s,r,d) in edgesNew if s == "GO:0015171" and d== "GO:0003333"]


   # print(testListOld)

  #  print(testListNew)
