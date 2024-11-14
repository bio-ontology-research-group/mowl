#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import Ontology, is_exp_code, is_cafa_target, FUNC_DICT

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot.pkl',
    help='Result file with a list of proteins, sequences and annotations')
@ck.option(
    '--org', '-org', type=ck.Choice(["mouse", "human"]), default="mouse",
    help='Organism')
def main(data_file, org):
    df = pd.read_pickle(data_file)

    if org == 'mouse':
        org_id = '559292'
        out_file = "data/4932.annotations.tsv"
    elif org == 'human':
        org_id = '9606'
        out_file = "data/9606.annotations.tsv"
    else:
        raise ValueError(f"Organism {org} not supported")
    
    f = open(out_file, 'w')
    for row in df.itertuples():
        if row.orgs != org_id:
            continue
        for st_id in row.string_ids:
            f.write(st_id)
            for go_id in row.exp_annotations:
                f.write('\t' + go_id)
            f.write('\n')
    f.close()


if __name__ == '__main__':
    main()
