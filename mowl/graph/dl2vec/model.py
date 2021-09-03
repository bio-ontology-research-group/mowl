

from mowl.model import model
import generate_graph as gen

class DL2Vec(GraphGenModel):
    def __init__(self, dataset):
        super.__init__(dataset)

    def parse(self, format='networkx'):

        ont = Ont(ontology_file, "elk")
        ont.processOntology()

        G = nx.Graph()


        # the restriction are min,max,exactly,some,only

        # there are conjunction or disjunction
        axiom_orig = ont.axiom_orig


        for line in axiom_orig:
            result = convert_graph(line.strip())

            # print("-"*40)
            for entities in result:

                G.add_edge(entities[0].strip(), entities[2].strip())
                G.edges[entities[0].strip(), entities[2].strip()]["type"] = entities[1].strip()
                G.nodes[entities[0].strip()]["val"] = False
                G.nodes[entities[2].strip()]["val"] = False

        with open(annotation_file, "r") as f:
            for line in f.readlines():
                entities = line.split()
                G.add_edge(entities[0].strip(), entities[1].strip())
                G.edges[entities[0].strip(), entities[1].strip()]["type"] = "HasAssociation"
                G.nodes[entities[0].strip()]["val"] = False
                G.nodes[entities[1].strip()]["val"] = False

        if format=='dgl':
            G = dgl.from_networkx(G)
        elif format!='networkx':
            raise Exception("Graph formats only support DGL and Networkx")

        return G
        
