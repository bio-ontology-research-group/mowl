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

    entities = set([c for c,_ in ontology_edges]) | set([d for c, d in ontology_edges])
    
    closure_edges = []
    with open(closure_edges_file, "r") as f:
        for line in f:
            line = line.rstrip("\n").split("\t")
            if line[0] in entities and line[1] in entities:
                closure_edges.append((line[0], line[1]))

    logging.info("Generating training negatives of type 1")
    train_negatives_1 = list()
    with ck.progressbar(ontology_edges) as bar:
        for c, d in bar:
            train_negatives_1.append((d,c))

    logging.info("Generating closure negatives of type 1")
    closure_negatives_1 = list()
    with ck.progressbar(closure_edges) as bar:
        for c, d in bar:
            closure_negatives_1.append((d,c))

        
    train_negatives_2 = list()
    forbidden_set = set(ontology_edges) | set(closure_edges) | set(train_negatives_1) | set(closure_negatives_1)
    
    list_entities = list(entities)
    logging.info("Generating training negatives of type 2")
    with ck.progressbar(ontology_edges) as bar:
        
        for c, d in bar:
            found = False

            while not found:
                d_ = random.choice(list_entities)
                if d_ in entities:
                    if not (c,d_) in forbidden_set:
                        train_negatives_2.append((c, d_))
                        found = True


    closure_negatives_2 = list()
    forbidden_set |= set(train_negatives_2)

    logging.info("Generating closure negatives of type 2")
    
    with ck.progressbar(closure_edges) as bar:

        for d, c in bar:
            found = False
            while not found:
                d_ = random.choice(list_entities)
                if d_ in entities:
                    if not (c, d_) in forbidden_set:
                        closure_negatives_2.append((c, d_))
                        found = True
                
    
    assert len(ontology_edges) == len(train_negatives_1)
    assert len(ontology_edges) == len(train_negatives_2)
    train_set = zip(ontology_edges, train_negatives_1, train_negatives_2)


    assert len(closure_edges) == len(closure_negatives_1), f"{len(closure_edges)}, {len(closure_negatives_1)}"
    assert len(closure_negatives_1) == len(closure_negatives_2)
    closure_set = list(zip(closure_edges, closure_negatives_1, closure_negatives_2))
    random.shuffle(closure_set)
    num_closure = len(closure_set) // 2
    valid_set = closure_set[:num_closure]
    test_set = closure_set[num_closure:]

    valid_size = len(ontology_edges) *20 //100
    

    valid_set = valid_set[:valid_size]
    test_set = test_set[:valid_size]
    
    
    with open("data/train_data.pkl", "wb") as f:
        pkl.dump(train_set, f)

    with open("data/valid_data.pkl", "wb") as f:
        pkl.dump(valid_set, f)

    with open("data/test_data.pkl", "wb") as f:
        pkl.dump(test_set, f)



if __name__ == "__main__":
    root = "data/"
    train_edges_file = root +  "train_pos_data.tsv"
    closure_edges_file = root + "only_closure_pos_data.tsv"

    get_splits(train_edges_file, closure_edges_file)
