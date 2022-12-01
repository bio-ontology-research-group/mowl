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
    '--out-file', '-of', default='data/annotations.tsv',
    help='Result file with annotations')
def main(data_file, out_file):
    df = pd.read_pickle(data_file)

    f = open(out_file, 'w')
    for row in df.itertuples():
        if row.orgs != '559292':
            continue
        for st_id in row.string_ids:
            f.write(st_id)
            for go_id in row.exp_annotations:
                f.write('\t' + go_id)
            f.write('\n')
    f.close()


if __name__ == '__main__':
    main()
