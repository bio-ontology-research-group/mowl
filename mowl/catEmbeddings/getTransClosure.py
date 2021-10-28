import sys

sys.path.insert(0, '')
sys.path.append('../../')

from mowl.datasets.base import PathDataset
from mowl.graph.taxonomy.model import TaxonomyParser
def getClosure():
    dataset = PathDataset("data/goslim_yeast.owl", "", "")

    parser = TaxonomyParser(dataset)

    parser.parserTrainSet.transitiveClosureToFile("data/trans_yeast.owl")


if __name__ == "__main__":
    getClosure()
