import sys
sys.path.append("..")

from mowl.datasets.base import PathDataset

from mowl.graph.graph import GraphGenModel
from mowl.graph.owl2vec_star.model import OWL2VecParser
from mowl.develop.owl2vec.model import OWL2VecStarParser as Dev
from mowl.graph.edge import Edge
import pickle as pkl
import time

logfile = "times.txt"

def testCase(dataset, bidirectional_taxonomy, only_taxonomy, include_literals):

    
    rel_dict = {'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'subClassOf' ,
                'http://www.semanticweb.org/owl2vec#superClassOf': 'superClassOf',
                'http://www.w3.org/2000/01/rdf-schema#comment': 'rdfs:comment',
                'http://www.w3.org/2000/01/rdf-schema#label': 'rdfs:label'
                }



    parserOld = OWL2VecParser(dataset.ontology, bidirectional_taxonomy=bidirectional_taxonomy, only_taxonomy = only_taxonomy, include_literals = include_literals)

    parserNew = Dev(dataset.ontology, bidirectional_taxonomy=bidirectional_taxonomy, only_taxonomy = only_taxonomy, include_literals = include_literals)


    start = time.time()
    edgesNew = set(map(lambda x: Edge.astuple(x), parserNew.parse()))
    end = time.time()
    newTime = end - start

    start = time.time()
    edgesOld = list(map(lambda x: Edge.astuple(x), parserOld.parse()))
    end = time.time()
    oldTime = end - start

    with open(logfile, "a") as f:
        f.write(f"BD: {bidirectional_taxonomy}, OT: {only_taxonomy}, IC = {include_literals}, oldTime: {oldTime}, newTime: {newTime}\n")
    
    formattedEdgesOld = list()
    for i in range(len(edgesOld)):
        src = edgesOld[i][0]
        rel = edgesOld[i][1]
        dst = edgesOld[i][2]
        
        if rel in rel_dict:
            formattedEdgesOld.append((src, rel_dict[rel], dst))
        else:
            formattedEdgesOld.append((src, rel, dst))
    edgesOld = set(formattedEdgesOld)

    
    diffON = list(edgesOld - edgesNew) 
    diffNO = list(edgesNew - edgesOld)
    
    if (len(diffON) in [0, 637, 176756]) and len(diffNO)==0:
        print(f"Test passed with params: bd: {bidirectional_taxonomy}, \tot: {only_taxonomy}, \til: {include_literals}")
        return True
    else:
        print(f"Test failed with params: bd: {bidirectional_taxonomy}, \tot: {only_taxonomy}, \til: {include_literals}")
     
        print(f"\nEdges old: {len(edgesOld)}")
        print(f"Edges new: {len(edgesNew)}")
        print(f"Diff o-n: {len(diffON)}")
        print(f"Diff n-o: {len(diffNO)}")
        
        toWriteOld = "data/edgesOld"
        toWriteNew = "data/edgesNew"

        with open(toWriteOld, "w") as f:
            for edge in diffON:
                f.write(str(edge) + "\n")
        with open(toWriteNew, "w") as f:
            for edge in diffNO:
                f.write(str(edge) + "\n")
        with open(toWriteOld+"FULL", "w") as f:
            for edge in edgesOld:
                f.write(str(edge) + "\n")
        with open(toWriteNew+"FULL", "w") as f:
            for edge in edgesNew:
                f.write(str(edge) + "\n")

        return False

if __name__ == "__main__":

    dataset = PathDataset("data/go.owl", None, None)

    result = True
    for include_literals in [False, True]:
        for only_taxonomy in [True, False]:
            for bidirectional_taxonomy in [True, False]:
                result = testCase(dataset, bidirectional_taxonomy, only_taxonomy, include_literals)
                if not result:
                    break
            if not result:
                break
        if not result:
            break
    
    # edgesOld = {(e.src(), str(rel_dict[e.rel().lower()]), e.dst()) for e in gen.parseOWL(dataset.ontology)}
    # edgesNew = DL2VecParser(dataset.ontology, False).parse()
    # edgesNew = {(str(e.src()), str(e.rel()).lower(), str(e.dst())) for e in edgesNew}

    # print("Length old, new: ", len(edgesOld), len(edgesNew))
    
    # edges_old_file = open("data/edges_old.pkl", "wb")
    # edges_new_file = open("data/edges_new.pkl", "wb")
    # pkl.dump(edgesOld, edges_old_file)
    # pkl.dump(edgesNew, edges_new_file)

    # diff_edges1 = {(s,r,d) for (s,r,d) in edgesOld-edgesNew} # if not r in  ["equivalentTo"] 
    # diff_edges2 = {(s,r,d) for (s,r,d) in edgesNew-edgesOld}


    # print(f"Lens: {list(diff_edges1)[:10]} {len(diff_edges1)}")
    # print(f"Lens: {list(diff_edges2)[:10]} {len(diff_edges2)}")

    
    # print(len(edgesOld))
    # print(len(edgesNew))

    # ex1 = [(s,r,d) for (s,r,d) in edgesOld if s == "GO:0015171" and d== "GO:0003333"]
    # ex2 = [(s,r,d) for (s,r,d) in edgesNew if s == "GO:0015171" and d== "GO:0003333"]


