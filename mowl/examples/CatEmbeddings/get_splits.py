import random
import pickle as pkl
import click as ck
import logging
logging.basicConfig(level = logging.INFO)

def get_splits(train_edges_file, closure_edges_file):
    ontology_edges = []
    with open(train_edges_file, "r") as f:
        for line in f:
            line = line.rstrip("\n").split("\t")
            ontology_edges.append((line[0], line[1]))

    entities = list(set([c for c,_ in ontology_edges]) | set([d for c, d in ontology_edges]))

    for e in entities:
        ontology_edges.append((e,e))
        
    must_be_entities = set(entities[:])

    closure_edges = []
    with open(closure_edges_file, "r") as f:
        for line in f:
            line = line.rstrip("\n").split("\t")
            c, d = line[0], line[1]
            if c in must_be_entities and d in must_be_entities:
                closure_edges.append((line[0], line[1]))

    

    
    logging.info("Generating training negatives of type 1")
    train_negatives_1 = list()
    with ck.progressbar(ontology_edges) as bar:
        for c, d in bar:
            train_negatives_1.append((d,c))

            
    closure_negatives_1 = list()
    logging.info("Generating testing negatives of type 1")
    
    with ck.progressbar(closure_edges) as bar:
        for c, d in bar:
            closure_negatives_1.append((d,c))

        
    train_negatives_2 = list()
    forbidden_set = set(ontology_edges) | set(closure_edges) | set(train_negatives_1) | set(closure_negatives_1)
    

    logging.info("Generating training negatives of type 2")
    with ck.progressbar(ontology_edges) as bar:
        for c, d in bar:
            found = False

            while not found:
                d_ = random.choice(entities)
                if d_ in must_be_entities:
                    if not (c,d_) in forbidden_set:
                        train_negatives_2.append((c, d_))
                        found = True


    closure_negatives_2 = list()
    forbidden_set |= set(train_negatives_2)
    logging.info("Generating testing negatives of type 2")
    with ck.progressbar(entities) as bar:
        for c in bar:
            for d in entities:
                if not (c, d) in forbidden_set:
                    closure_negatives_2.append((c, d))
                    
                
    
    assert len(ontology_edges) == len(train_negatives_1)
    assert len(ontology_edges) == len(train_negatives_2)
    train_set = [(e,1) for e in ontology_edges]
    train_set += [(e,0) for e in train_negatives_1]
    train_set += [(e,0) for e in train_negatives_2]


    test_set = [(e,1) for e in closure_edges]
    test_set += [(e,0) for e in closure_negatives_1]
    test_set += [(e,0) for e in closure_negatives_2]
    
    with open(root + "train_data.pkl", "wb") as f:
        pkl.dump(train_set, f)

    with open(root + "test_data.pkl", "wb") as f:
        pkl.dump(test_set, f)



if __name__ == "__main__":
    root = "data/go/"
    train_edges_file = root +  "train_pos_data.tsv"
    closure_edges_file = root + "only_closure_pos_data.tsv"

    get_splits(train_edges_file, closure_edges_file)
