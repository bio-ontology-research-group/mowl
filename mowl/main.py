#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging

from rdflib import Graph

@ck.command()
@ck.option('--data-root', '-dr', default='data/', help='Data root folder', required=True)
def main(data_root):
    # TODO: tool entrypoint starts here
    

if __name__ == '__main__':
    main()
