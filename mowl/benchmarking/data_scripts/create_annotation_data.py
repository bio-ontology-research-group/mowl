import sys
sys.path.append('../../../')
import os
import random
from math import floor

import logging

root = "data/"
root_mouse = "data_mouse/"
root_human = "data_human/"

def gene_phen_annots(verbose = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    in_file = root + 'HMD_HumanPhenotype.rpt'
    out_file = root + 'gene_annots.tsv'

    with open(out_file, 'w') as fout:
        with open(in_file, 'r') as fin:
            for line in fin:
                line = line.strip().split('\t')
                if len(line) < 5:
                    continue
                h_gene, id_gene, m_gene, mgi, phen_annots = tuple(line)
                phen_annots = phen_annots.split(", ")

                id_gene = "http://" + id_gene
                phen_annots = list(map(lambda x: "http://purl.obolibrary.org/obo/" + x.replace(":", "_"), phen_annots))
                out = "\t".join([id_gene] + phen_annots)
                fout.write(out+"\n")


def disease_phen_annots(verbose = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    in_file = root + 'phenotype.hpoa'
    out_file = root + 'disease_annots.tsv'

    with open(out_file, 'w') as fout:
        with open(in_file, 'r') as fin:
            for line in fin:
                if line.startswith("#"):
                    continue
                
                line = line.strip().split('\t')

                disease_id = line[0]
                phenotype = line[3]

                out = "http://" + disease_id + "\t" + "http://purl.obolibrary.org/obo/"+phenotype.replace(":", "_")
                fout.write(out+"\n")

def gene_disease_assoc(species, root_sp):

    in_file = root + "MGI_DO.rpt"
    out_file = root_sp + f"gene_disease_assoc_{species}.tsv"

    with open(out_file, "w") as fout:
        with open(in_file, "r") as fin:
            for line in fin:
                if line.startswith("DO "):
                    continue

                line = line.strip().split('\t')

                if not species in line[3]:
                    continue
                disease_id = line[2]
                gene_id = line[6]

                if gene_id != "" and disease_id != "":
                    diseases = disease_id.split("|")
                    for disease in diseases:
                        out = "http://"+gene_id + "\t" + "http://"+disease 
                        fout.write(out+"\n")
                

def split_associations(in_file, species, root_sp):

    train_file = root_sp + f"train_assoc_data_{species}.tsv"
    valid_file = root_sp + f"valid_assoc_data_{species}.tsv"
    test_file = root_sp + f"test_assoc_data_{species}.tsv"

    with open(in_file, "r") as fin:
        assocs = fin.readlines()

    random.shuffle(assocs)

    n_assocs = len(assocs)
    train_idx = floor(n_assocs*0.8)
    valid_idx = train_idx + floor(n_assocs*0.1)
    test_idx = valid_idx + floor(n_assocs*0.1)

    train_assocs = assocs[:train_idx]
    valid_assocs = assocs[train_idx:valid_idx]
    test_assocs = assocs[valid_idx:]

    with open(train_file, "w") as f:
        for line in train_assocs:
            f.write(line)

    with open(valid_file, "w") as f:
        for line in valid_assocs:
            f.write(line)

    with open(test_file, "w") as f:
        for line in test_assocs:
            f.write(line)


def create_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
                        
if __name__ == "__main__":

    create_dir(root)
    create_dir(root_human)
    create_dir(root_mouse)
    
    gene_phen_annots()
    disease_phen_annots()

    
    species = ["mouse", "human"]

    for sp in species:
        if sp == "mouse":
            root_sp = root_mouse
        elif sp == "human":
            root_sp = root_human
        else:
            raise ValueError("Species name not recognized")

        in_file = root_sp + f"gene_disease_assoc_{sp}.tsv"

        gene_disease_assoc(sp, root_sp)
        split_associations(in_file, sp, root_sp)
