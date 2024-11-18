import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    
from .edge import Edge, Node
from .utils import IGNORED_AXIOM_TYPES, IGNORED_EXPRESSION_TYPES, pairs

from mowl.projection.base import ProjectionModel
from mowl.owlapi import OWLAPIAdapter
from mowl.owlapi.defaults import BOT, TOP
import mowl.error.messages as msg
from mowl.projection import Edge as mEdge

from org.semanticweb.owlapi.model import  AxiomType, EntityType, OWLObjectInverseOf, OWLOntology
from org.semanticweb.owlapi.model import ClassExpressionType as CT

import networkx as nx
from deprecated.sphinx import versionadded

from tqdm import tqdm

adapter = OWLAPIAdapter()

top_node = Node(owl_class = adapter.create_class(TOP))
bot_node = Node(owl_class = adapter.create_class(BOT))

class Graph():
    def __init__(self, abox_edges = None):
        self._node_to_id = {}
        self._id_to_node = {}
        self._out_edges = dict()
        self._in_edges = dict()

        if abox_edges is None:
            self.abox_edges = []
        else:
            self.abox_edges = abox_edges
        
    @property
    def node_to_id(self):
        return self._node_to_id

    @property
    def id_to_node(self):
        return self._id_to_node
    
    @property
    def nodes(self):
        return set(self.node_to_id.keys())

    @property
    def out_edges(self):
        return self._out_edges

    @property
    def in_edges(self):
        return self._in_edges

    def add_node(self, node):

        if node in self.node_to_id:
            return
        
        if not isinstance(node, Node):
            raise TypeError(f"Node must be of type Node. Got {type(node)}")

        if node.is_owl_thing():
            return
        if node.is_owl_nothing():
            return
        
        if bot_node not in self.node_to_id:
            self.node_to_id[bot_node] = len(self.node_to_id)
            self.id_to_node[self.node_to_id[bot_node]] = bot_node
            self.out_edges[bot_node] = set()
            self.out_edges[bot_node].add(top_node)
            self.in_edges[bot_node] = set()

        if top_node not in self.node_to_id:
            self.node_to_id[top_node] = len(self.node_to_id)
            self.id_to_node[self.node_to_id[top_node]] = top_node
            self.out_edges[top_node] = set()
            self.in_edges[top_node] = set()
            self.in_edges[top_node].add(bot_node)

            
        if not node in self.node_to_id:
            self.node_to_id[node] = len(self.node_to_id)
            self.id_to_node[self.node_to_id[node]] = node
            self.out_edges[node] = set()
            self.in_edges[node] = set()

        self.in_edges[node].add(node)
        self.out_edges[node].add(node)

        self.in_edges[node].add(bot_node)
        self.out_edges[bot_node].add(node)

        self.in_edges[top_node].add(node)
        self.out_edges[node].add(top_node)


                    
        if node.in_object_category() and not node.domain and not node.codomain and node.owl_class.getClassExpressionType() == CT.OWL_CLASS:
            negated = node.negate()
            and_node = adapter.create_object_intersection_of(node.owl_class, negated.owl_class)
            and_node = Node(owl_class = and_node)
            self.add_edge(Edge(and_node, "http://arrow", bot_node))
            self.add_edge(Edge(and_node, "http://arrow", node))
            self.add_edge(Edge(and_node, "http://arrow", negated))

            or_node = adapter.create_object_union_of(node.owl_class, negated.owl_class)
            or_node = Node(owl_class = or_node)
            self.add_edge(Edge(top_node, "http://arrow", or_node))
            self.add_edge(Edge(node, "http://arrow", or_node))
            self.add_edge(Edge(negated, "http://arrow", or_node))
            
        self.add_node(node.nnf())
                                        
            
    def add_edge(self, edge):
        src = edge.src
        dst = edge.dst
        self.add_node(src)
        self.add_node(dst)
                                                    
        self.out_edges[src].add(dst)
        self.in_edges[dst].add(src)

    def add_all_edges(self, *edges):
        for edge in edges:
            self.add_edge(edge)

    def as_edgelist(self):
        edges = []
        for source, targets in self.out_edges.items():
            for target in targets:
                edges.append((source, target))
        return edges

    def as_str_edgelist(self):
        edges = []
        for source, targets in self.out_edges.items():
            source = str(source)
            for target in targets:
                target = str(target)
                edges.append((source, "http://arrow", target))
        for source, target in self.abox_edges:
            source = str(source)
            target = str(target)
            edges.append((source, "http://type", target))

        edges  = [mEdge(*edge) for edge in edges]
        return edges

    def as_edges(self):
        for source, targets in self.out_edges.items():
            for target in targets:
                yield Edge(source, "http://arrow", target)

        for source, target in self.abox_edges:
            yield Edge(source, "http://type", target)
            
    def _lemma_6(self):
        # First equation, nothing to do
        # Second equation and left side of third equation and left side of fourth equation
        negated_nodes = set()
        for node in self.nodes:
            if node.is_negated() and not node.negated_domain:
                negated_nodes.add(node)
            
        for neg_node in tqdm(negated_nodes, desc="Lemma 6: Processing negated nodes"):
            in_edges = list(self.in_edges[neg_node])
            for in_node in in_edges:
                if in_node.is_owl_nothing() or in_node.is_owl_thing():
                    continue
                if in_node.domain:
                    continue

                node = neg_node.get_operand()
                
                if node == in_node:
                    continue
                
                neg_in_node = in_node.negate()
                op_edge = Edge(node, "saturation_lemma6", neg_in_node)

                intersection_owl = adapter.create_object_intersection_of(node.owl_class, in_node.owl_class)
                if len(intersection_owl.getNNF().getOperandsAsList()) == 1:
                    continue

                intersection = Node(owl_class = intersection_owl)
                
                assert node.domain == intersection.domain, f"Domain of {node} is {node.domain} and domain of {intersection} is {intersection.domain}"
                assert node.codomain == intersection.codomain, f"Codomain of {node} is {node.codomain} and codomain of {intersection} is {intersection.codomain}"
                assert node.negated_domain == intersection.negated_domain, f"Negated domain of {node} is {node.negated_domain} and negated domain of {intersection} is {intersection.negated_domain}"
                edge_int = Edge(intersection, "saturation_lemma6", bot_node)

            out_edges = list(self.out_edges[neg_node])
            for out_node in out_edges:
                if out_node.is_owl_thing() or out_node.is_owl_nothing():
                    continue
                if out_node.domain:
                    continue
                
                node = neg_node.get_operand()
                if node == out_node:
                    continue
                

                union = adapter.create_object_union_of(node.owl_class, out_node.owl_class)
                if len(union.getNNF().getOperandsAsList()) == 1:
                    continue
                
                union = Node(owl_class = union)
                edge_un = Edge(top_node, "saturation_lemma6", union)

                self.add_all_edges(op_edge, edge_int, edge_un)

                #TODO last equation (Morgan law)

        # Left side of third equation
        intersection_nodes = set()
        
        for node in self.nodes:
            if node.is_intersection():
                intersection_nodes.add(node)

        for int_node in tqdm(intersection_nodes, desc="Lemma 6: Processing intersection nodes"):
            if not bot_node in self.out_edges[int_node]:
                continue

            operands = list(int_node.owl_class.getOperandsAsList())
            intersection_pairs = pairs(operands)
            
            for node1, node2 in intersection_pairs:
                node1 = [n for n in node1 if not n.isOWLThing()] if len(node1) > 1 else node1
                node2 = [n for n in node2 if not n.isOWLThing()] if len(node2) > 1 else node2
                if len(node1) > 1:
                    node1 = adapter.create_object_intersection_of(*node1)
                else:
                    node1 = node1[0]
                node1 = Node(owl_class = node1)
                if len(node2) > 1:
                    node2 = adapter.create_object_intersection_of(*node2)
                else:
                    node2 = node2[0]
                node2 = Node(owl_class = node2.getObjectComplementOf())
                edge = Edge(node1, "saturation_lemma6", node2)
                self.add_edge(edge)
                
        # Left side of fourth equation
        union_nodes = set()
        for node in self.nodes:
            if node.is_union():
                
                union_nodes.add(node)

        for un_node in tqdm(union_nodes, desc="Lemma 6: Processing union nodes"):
            
            if not top_node in self.in_edges[un_node]:
                continue

            operands = un_node.owl_class.getOperandsAsList()
            union_pairs = pairs(operands)
            for node1, node2 in union_pairs:
                node1 = [n for n in node1 if not n.isOWLNothing()] if len(node1) > 1 else node1
                node2 = [n for n in node2 if not n.isOWLNothing()] if len(node2) > 1 else node2
                if len(node1) > 1:
                    node1 = adapter.create_object_union_of(*node1)
                else:
                    node1 = node1[0]
                node1 = Node(owl_class = node1.getObjectComplementOf())

                if len(node2) > 1:
                    node2 = adapter.create_object_union_of(*node2)
                else:
                    node2 = node2[0]
                    
                node2 = Node(owl_class = node2)

                edge = Edge(node1, "saturation_lemma6", node2)
                self.add_edge(edge)
                        


    def _definition_6(self):
        # Def 6. Although it is not defined explicitely in the paper, this definition will look for classes that are subclass of a disjointness.
        for node in self.nodes:
            if node.in_relation_category():
                continue
            if node.is_owl_thing() or node.is_owl_nothing():
                continue
            
            node_to_neg = dict()
            for out_node in self.out_edges[node]:
                if out_node.domain or out_node.codomain:
                    continue
                if out_node.is_owl_thing() or out_node.is_owl_nothing():
                    continue

                
                if not out_node in node_to_neg and not out_node.negate() in node_to_neg:
                    node_to_neg[out_node] = None
                elif out_node in node_to_neg:
                    continue
                elif out_node.negate() in node_to_neg:
                    node_to_neg[out_node.negate()] = out_node
                    
            for node1, node2 in node_to_neg.items():
                if node2 is None:
                    continue
                else:
                    intersection = adapter.create_object_intersection_of(node1.owl_class, node2.owl_class)
                    intersection = Node(owl_class = intersection)
                    edge = Edge(intersection, "saturation_lemma6", bot_node)
                    self.add_edge(edge)
                    edge2 = Edge(node, "saturation_lemma6", intersection)
                    self.add_edge(edge2)
                                        
                            
    def _definition_7(self):
        #Last equation, other equations are covered in lemma 8
        relations = set()
        for node in self.nodes:
            if node.is_whole_relation():
                relations.add(node)

        for rel in tqdm(relations, desc="Definition 7: Processing relations"):
            for in_rel in self.in_edges[rel]:
                if not in_rel.in_relation_category():
                    continue
                in_rel_codomain = in_rel.to_codomain()
                if not in_rel_codomain in self.nodes:
                    continue
                 
                for cod in self.out_edges[in_rel_codomain]:
                    if cod.domain or cod.codomain or cod.in_relation_category():
                        continue
                    in_rel_domain = in_rel.to_domain()

                    ex_rel_cod = adapter.create_object_some_values_from(rel.relation, cod.owl_class)
                    node = Node(owl_class = ex_rel_cod, relation = rel.relation, domain = True)
                    edge = Edge(in_rel_domain, "saturation_definition7", node)
                    self.add_edge(edge)
                    
                
                    
    def _lemma_8(self):
        relations = set()
        for node in self.nodes:
            if node.is_whole_relation():
                relations.add(node)

        for node in tqdm(self.nodes, desc="Lemma 8: Processing nodes"):
            if not node.in_object_category():
                continue
            if node.is_intersection() or node.is_union() or node.is_existential():
                continue
            if node.domain or node.codomain:
                continue
            if node.is_owl_thing() or node.is_owl_nothing():
                continue

            out_edges = list(self.out_edges[node])

            for out_node in out_edges:
                if not out_node.in_object_category():
                    continue
                if out_node.is_intersection() or out_node.is_union() or out_node.is_existential():
                    continue
                if out_node.domain or out_node.codomain:
                    continue
                if out_node.is_owl_thing():
                    continue
                
                for relation in relations:
                    ex_class = adapter.create_object_some_values_from(relation.relation, node.owl_class)
                    ox_out_class = adapter.create_object_some_values_from(relation.relation, out_node.owl_class)
                    ex_node = Node(owl_class = ex_class)
                    ox_out_node = Node(owl_class = ox_out_class)
                    edge = Edge(node, "saturation_lemma8", ex_node)
                    self.add_edge(edge)
                
    def _lemma_8_bk(self):
        existential_nodes = set()
        for node in self.nodes:
            if node.is_existential():
                existential_nodes.add(node)

        for node in existential_nodes:
            filler = node.owl_class.getFiller()
            property_ = node.owl_class.getProperty()
            filler_node = Node(owl_class = filler)
            domain_node = Node(owl_class = node.owl_class, relation = property_, domain = True)


            in_edges = list(self.in_edges[filler_node])
            for in_filler in in_edges:
                ex_in_class = adapter.create_object_some_values_from(node.owl_class.getProperty(), in_filler.owl_class)
                ex_in_node = Node(owl_class = ex_in_class)
                edge = Edge(ex_in_node, "saturation_lemma8", node)
                self.add_edge(edge)

                domain_in_node = Node(owl_class=ex_in_class, relation = property_, domain = True)
                edge = Edge(domain_in_node, "saturation_lemma8", domain_node)
                self.add_edge(edge)
                edge = Edge(domain_in_node, "saturation_lemma8", ex_in_node)
                self.add_edge(edge)
                edge = Edge(ex_in_node, "saturation_lemma8", domain_in_node)
                self.add_edge(edge)

            out_edges = list(self.out_edges[filler_node])
            for out_filler in out_edges:
                if out_filler.owl_class is None:
                    continue
                if out_filler.is_owl_nothing():
                    continue
                    edge = Edge(node, "saturation_lemma8", bot_node)
                    self.add_edge(edge)
                else:
                    ex_out_class = adapter.create_object_some_values_from(node.owl_class.getProperty(), out_filler.owl_class)
                    ex_out_node = Node(owl_class = ex_out_class)
                    edge = Edge(node, "saturation_lemma8", ex_out_node)
                    self.add_edge(edge)

                    domain_out_node = Node(owl_class=ex_out_class, relation = property_, domain = True)
                    edge = Edge(domain_node, "saturation_lemma8", domain_out_node)
                    self.add_edge(edge)
                    edge = Edge(domain_out_node, "saturation_lemma8", ex_out_node)
                    self.add_edge(edge)
                    edge = Edge(ex_out_node, "saturation_lemma8", domain_out_node)
                    self.add_edge(edge)

        
    def as_nx(self):
        logging.debug("Converting to networkx")
        G = nx.DiGraph()
        G.add_nodes_from(list(self.node_to_id.values()))
        for edge in self.as_edgelist():
            src, dst = edge
                            
            if src.in_object_category() and dst.in_object_category():
                if src == bot_node or dst == top_node:
                    continue
                G.add_edge(self.node_to_id[src], self.node_to_id[dst])
        logging.debug("Done converting to networkx")
        return G
                
    def transitive_closure(self):
        logging.debug("Computing transitive closure")
        G = self.as_nx()
        G = nx.transitive_closure(G)
        logging.debug("Done computing transitive closure in NetworkX")
        for src, dst in tqdm(G.edges(), desc="Adding transitive closure edges to graph"):
            src = self.id_to_node[src]
            dst = self.id_to_node[dst]
            self.add_edge(Edge(src, "transitive_closure", dst))
        logging.debug("Done computing transitive closure")

                
    def saturate(self, def_6 = True, lemma_6 = True, def_7 = True, lemma_8 = True):
        """
        This function saturates the graph using the rules defined in [brieulle2022]_.

        :param def_6: If ``True``, apply definition 6. Default is ``True``.
        :type def_6: bool, optional
        :param lemma_6: If ``True``, apply lemma 6. Default is ``True``.
        :type lemma_6: bool, optional
        :param def_7: If ``True``, apply definition 7. Default is ``True``.
        :type def_7: bool, optional
        :param lemma_8: If ``True``, apply lemma 8. Default is ``True``.
        :type lemma_8: bool, optional
        """

        if def_6:
            self._definition_6()

        if lemma_6:
            self._lemma_6()

        if def_7:
            self._definition_7()

        if lemma_8:
            self._lemma_8()

    def is_unsatisfiable(self, node):
        if not isinstance(node, Node):
            raise TypeError("node must be of type Node")
        if node not in self.nodes:
            raise ValueError("Node is not in graph")
        out_edges = self.out_edges[node]
        if bot_node in out_edges:
            return True
        else:
            return False

    def get_unsatisfiable_nodes(self):

        def is_trivial_unsat(node):
            trivial = False
            if node.domain or node.codomain or node.in_relation_category():
                return trivial

            owl_class = node.owl_class
            if owl_class.getClassExpressionType() == CT.OBJECT_INTERSECTION_OF:
                ops = owl_class.getOperandsAsList()
                if len(ops) == 2:
                    op1, op2 = tuple(ops)
                    if op2.getClassExpressionType() == CT.OBJECT_COMPLEMENT_OF:
                        op2 = op2.getOperand()
                        if op1.equals(op2):
                            trivial = True
                    if op1.getClassExpressionType() == CT.OBJECT_COMPLEMENT_OF:
                        op1 = op1.getOperand()
                        if op2.equals(op1):
                            trivial = True
            return trivial
        
        unsat = set()
        for node in self._in_edges[bot_node]:
            if is_trivial_unsat(node):
                continue
            unsat.add(node)
            
        return unsat


    def write_to_file(self, outfile):
        with open(outfile, "w") as f:
            edges = self.as_str_edgelist()
            for src, rel, dst in tqdm(edges, desc="Writing to file"):
                f.write(f"{src}\t{rel}\t{dst}\n")

@versionadded(version="0.3.0")
class CategoricalProjector(ProjectionModel):
    """
    Implementation of projection rules defined in [zhapa2023]_.
    
    This class implements the projection of ALC axioms into a graph using categorical diagrams. Saturation steps can be applied following [brieulle2022]_.

    :param saturation_steps: Number of saturation steps of the graph/category.
    :type saturation_steps: int
    :param transitive_closure: If ``True``, every saturation step computes the transitive edges.
    :type transitive_closure: bool
    :param def_6: If ``True``, apply definition 6. Default is ``True``.
    :type def_6: bool, optional
    :param lemma_6: If ``True``, apply lemma 6. Default is ``True``.
    :type lemma_6: bool, optional
    :param def_7: If ``True``, apply definition 7. Default is ``True``.
    :type def_7: bool, optional
    :param lemma_8: If ``True``, apply lemma 8. Default is ``True``.
    :type lemma_8: bool, optional
    """

    def __init__(self, output_type, saturation_steps = 0, transitive_closure = False, def_6 = True, lemma_6 = True, def_7 = True, lemma_8 = True):


        if not isinstance(output_type, str):
            raise TypeError(msg.type_error("output_type", "str", type(output_type)))
        if not isinstance(saturation_steps, int):
            raise TypeError(msg.type_error("saturation_steps", "int", type(saturation_steps), optional=True))
        if saturation_steps < 0:
            raise ValueError("Optional parameter saturation_steps must be non-negative")
        if not isinstance(transitive_closure, bool):
            raise TypeError(msg.type_error("transitive_closure", "bool", type(transitive_closure), optional=True))
        if not isinstance(def_6, bool):
            raise TypeError(msg.type_error("def_6", "bool", type(def_6), optional=True))
        if not isinstance(lemma_6, bool):
              raise TypeError(msg.type_error("lemma_6", "bool", type(lemma_6), optional=True))
        if not isinstance(def_7, bool):
            raise TypeError(msg.type_error("def_7", "bool", type(def_7), optional=True))
        if not isinstance(lemma_8, bool):
            raise TypeError(msg.type_error("lemma_8", "bool", type(lemma_8), optional=True))

        
        if not output_type in ["str", "owl"]:
            raise ValueError("Invalid output type. Must be 'str' or 'owl'")

        
        self.output_type = output_type
        self.saturation_steps = saturation_steps
        self.transitive_closure = transitive_closure
        self.def_6 = def_6
        self.lemma_6 = lemma_6
        self.def_7 = def_7
        self.lemma_8 = lemma_8

        self.adapter = OWLAPIAdapter()
        self.ont_manager = self.adapter.owl_manager
        self.data_factory = self.adapter.data_factory
        self.graph = Graph()


        
    def project(self, ontology):
        r"""Generates the projection of the ontology.

        :param ontology: The ontology to be processed.
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
        """
        

        if not isinstance(ontology, OWLOntology):
            raise TypeError(
                "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology")

        
        all_classes = ontology.getClassesInSignature()
        #get class assertion axioms
        abox_edges = []
        for cls in all_classes:
            cls_str = str(cls.toStringID())
            cls_assertions = list(ontology.getClassAssertionAxioms(cls))
            for axiom in cls_assertions:
                ind = str(axiom.getIndividual().toStringID())
                abox_edges.append((ind, cls_str))
        
        self.graph = Graph(abox_edges = abox_edges)
        
        
        for cls in tqdm(all_classes, desc="Adding nodes to graph"):
            self.graph.add_node(Node(owl_class = cls))
        all_axioms = ontology.getAxioms(True)
        
        for axiom in tqdm(all_axioms, total = len(all_axioms), desc="Processing axioms"):
            self.graph.add_all_edges(*list(self.process_axiom(axiom)))

        if self.saturation_steps > 0:
            for i in range(self.saturation_steps):
                self.graph.saturate(def_6 = self.def_6, lemma_6 = self.lemma_6, def_7 = self.def_7, lemma_8 = self.lemma_8)
                if self.transitive_closure:
                    self.graph.transitive_closure()

        if self.output_type == "str":
            return self.graph.as_str_edgelist()
        elif self.output_type == "owl":
            return self.graph.as_edges()
        else:
            raise ValueError("Invalid output type. Must be 'str' or 'owl'")
        
    def process_axiom(self, axiom):
        """Process an OWLAxiom and return a list of edges.

        :param axiom: Axiom to be parsed.
        :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
        """

        axiom_type = axiom.getAxiomType()
        if axiom_type == AxiomType.SUBCLASS_OF:
            return self._process_subclassof(axiom)
        elif axiom_type == AxiomType.EQUIVALENT_CLASSES:
            return self._process_equivalent_classes(axiom)
        elif axiom_type == AxiomType.DISJOINT_CLASSES:
            return self._process_disjointness(axiom)
        elif axiom_type == AxiomType.CLASS_ASSERTION:
            return self._process_class_assertion(axiom)
        elif axiom_type == AxiomType.OBJECT_PROPERTY_ASSERTION:
            return self._process_object_property_assertion(axiom)
        elif axiom_type in IGNORED_AXIOM_TYPES:
            #Ignore these types of axioms
            return []
        else:
            print(f"process_axiom: Unknown axiom type: {axiom_type}")
            return []


    def _process_equivalent_classes(self, axiom):
        """Process an EquivalentClasses axiom and return a list of edges."""

        subclass_axioms = axiom.asOWLSubClassOfAxioms()
        edges = set()
        for subclass_axiom in subclass_axioms:
            edges |= self._process_subclassof(subclass_axiom)
        
        if edges == set():
            print(f"No edges found for EquivalentClasses axiom: {axiom}")
        return edges

    def _process_disjointness(self, axiom):
        """Process a disjointness axiom"""
        subclass_axioms = axiom.asOWLSubClassOfAxioms()
        edges = set()
        for subclass_axiom in subclass_axioms:
            edges |= self._process_subclassof(subclass_axiom)
        
        return edges

    def _process_class_assertion(self, axiom):
        """Process a class assertion axiom"""
        individual = axiom.getIndividual()
        class_expression = axiom.getClassExpression()

        node, edges = self._process_expression_and_get_complex_node(class_expression)
        ind_as_class = adapter.create_class(str(individual.toStringID()))
        ind_node = Node(owl_class = ind_as_class, is_individual=True)
        edges.add(self._assertion_arrow(ind_node, node))
        
        return edges

    def _process_object_property_assertion(self, axiom):
        """Process an object property assertion axiom"""
        subject = axiom.getSubject()
        object_ = axiom.getObject()
        property_ = axiom.getProperty()

        subject_as_class = adapter.create_class(str(subject.toStringID()))
        object_as_class = adapter.create_class(str(object_.toStringID()))
        subject_node = Node(owl_class = subject_as_class, is_individual=True)
        object_node = Node(owl_class = object_as_class, is_individual=True)
        
        domain_node = Node(relation=property_, domain=True)
        codomain_node = Node(relation=property_, codomain=True)

        some_values_from = adapter.create_object_some_values_from(property_, object_as_class)
        
        edges = set()
        edges.add(self._subsumption_arrow(subject_node, domain_node))
        edges.add(self._subsumption_arrow(object_node, codomain_node))
        edges.add(self._subsumption_arrow(subject_node, some_values_from))
        
        return edges
        
    def _process_subclassof(self, axiom):
        """Process a SubClassOf axiom and return a list of edges."""
        sub_class = axiom.getSubClass()
        super_class = axiom.getSuperClass()
        sub_edges = set()
        super_edges = set()

        sub_node, sub_edges = self._process_expression_and_get_complex_node(sub_class)
        #print(axiom)
        super_node, super_edges = self._process_expression_and_get_complex_node(super_class)
                                
        if (sub_node is None) or (super_node is None):
            return set()

        edges = set()
        edges.add(self._subsumption_arrow(sub_node, super_node))
        edges |= sub_edges
        edges |= super_edges

        not_sub_class = self.adapter.create_complement_of(sub_class)
        union = self.adapter.create_object_union_of(not_sub_class, super_class)
        union_complex_node, union_edges = self._process_expression_and_get_complex_node(union)

        
        
        edges |= union_edges
        edges.add(self._subsumption_arrow(top_node, union_complex_node))
        
        return edges


    def _process_expression_and_get_complex_node(self, expression):

        expr_type = expression.getClassExpressionType()

        edges = set()
        
        if expr_type == CT.OWL_CLASS:
            return Node(owl_class=expression), edges
            
        if expr_type == CT.OBJECT_INTERSECTION_OF:
            prod_complex_node = expression
            operands = expression.getOperandsAsList()
                        
            for op in operands:
                op_complex_node, op_edges = self._process_expression_and_get_complex_node(op)
                if op_complex_node is None:
                    return None, set()

                edges |= op_edges
                edges.add(self._product_limit_arrow(prod_complex_node, op_complex_node))

            #TODO: add arrows to fulfill distributivity

            return prod_complex_node, edges
            
        elif expr_type == CT.OBJECT_UNION_OF:
            coprod_complex_node = expression
            operands = expression.getOperandsAsList()
            
            for op in operands:
                op_complex_node, op_edges = self._process_expression_and_get_complex_node(op)
                if op_edges == None:
                    return None, set()
                
                edges |= op_edges
                edges.add(self._coproduct_limit_arrow(op_complex_node,
                                                     coprod_complex_node))

                                                                            
            return coprod_complex_node, edges
                    
        elif expr_type == CT.OBJECT_SOME_VALUES_FROM:
            
            existential_complex_node = expression
                            
            property_ = expression.getProperty()

            if not isinstance(property_, OWLObjectInverseOf):
                if property_.getEntityType() != EntityType.OBJECT_PROPERTY:
                    return None, set()
            else:
                if property_.isNamed():
                    property_ = property_.asOWLObject()
                else:
                    return None, set()
                    
            filler = expression.getFiller()
            filler_info = self._process_expression_and_get_complex_node(filler)
            filler_node, filler_edges = filler_info
            if filler_node is None:
                return None, set()
            
            edges |= filler_edges


            rel_exists_r_c = Node(owl_class = expression, relation=property_)
            codomain_rel_exists_r_c = Node(owl_class=expression, relation=property_, codomain=True)
            domain_rel_exists_r_c = Node(owl_class=expression, relation=property_, domain=True)
                        
            edges.add(Edge(rel_exists_r_c, "http://general_arrow", Node(relation=property_)))
            edges.add(Edge(codomain_rel_exists_r_c, "http://general_arrow", Node(owl_class=filler)))
            
            exist_node = Node(owl_class=existential_complex_node)
            edges.add(Edge(domain_rel_exists_r_c, "http://general_arrow", exist_node))
            edges.add(Edge(exist_node, "http://general_arrow", domain_rel_exists_r_c))
                        
            return existential_complex_node, edges
            
                         
        elif expr_type == CT.OBJECT_ALL_VALUES_FROM:
            universal_complex_node = expression
                                        
            property_ = expression.getProperty()
            filler = expression.getFiller()
            not_filler = self.adapter.create_complement_of(filler)
            rel_not_filler = self.adapter.create_object_some_values_from(property_, not_filler)
            not_rel_not_filler = self.adapter.create_complement_of(rel_not_filler)

            not_rel_not_filler_info = self._process_expression_and_get_complex_node(not_rel_not_filler)
            not_rel_not_filler_node, not_rel_not_filler_edges = not_rel_not_filler_info
            if not_rel_not_filler_node is None:
                return None, set()

            filler_info = self._process_expression_and_get_complex_node(filler)
            filler_node, filler_edges = filler_info
            if filler_node is None:
                return None, set()
            
            edges |= not_rel_not_filler_edges
            edges |= filler_edges

            edges.add(self._subsumption_arrow(universal_complex_node, not_rel_not_filler_node))
            edges.add(self._subsumption_arrow(not_rel_not_filler_node, universal_complex_node))

            not_domain_rel_n_filler = Node(owl_class=rel_not_filler, relation=property_, domain=True, negated_domain=True)
            edges.add(self._subsumption_arrow(not_domain_rel_n_filler, universal_complex_node))
            edges.add(self._subsumption_arrow(universal_complex_node, not_domain_rel_n_filler))
            return universal_complex_node, edges

        elif expr_type == CT.OBJECT_COMPLEMENT_OF:
            
            negation_complex_node = expression
            operand = expression.getOperand()
            
            operand_info = self._process_expression_and_get_complex_node(operand)
            operand_node, operand_edges = operand_info

            union = self.adapter.create_object_union_of(expression, operand)
            union = Node(owl_class = union)
            intersection = self.adapter.create_object_intersection_of(expression, operand)
            intersection = Node(owl_class=intersection)
            edges |= operand_edges

                                                
            edges.add(Edge(intersection, "http://general_arrow", bot_node))
            edges.add(Edge(top_node, "http://general_arrow", union))

            return negation_complex_node, edges

        elif expr_type in IGNORED_EXPRESSION_TYPES:
            return None, set()
        else:
            print("process expression and get complex node: Unknown super class type: {}".format(expression))
        
                            
    ####################### MORPHISMS ###########################

    # decorator that transforms params as Node
    def _node_params(func):
        def wrapper(self, src, dst):
            if not isinstance(src, Node):
                src = Node(owl_class=src)
            if not isinstance(dst, Node):
                dst = Node(owl_class=dst)
            return func(self, src, dst)
        return wrapper

    @_node_params
    def _assertion_arrow(self, src, dst):
        rel = "http://type_arrow"
        return Edge(src, rel, dst)

    @_node_params
    def _subsumption_arrow(self, src, dst):
        rel = "http://subsumption_arrow"
        return Edge(src, rel, dst)

    @_node_params
    def _coproduct_limit_arrow(self, src, dst):
        rel = "http://injects"
        return Edge(src, rel, dst)

    @_node_params
    def _product_limit_arrow(self, src, dst):
        rel = "http://projects"
        return Edge(src, "http://projects", dst)

def add_extra_existential_axioms(ontology):
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    
    classes = ontology.getClassesInSignature()
    roles = ontology.getObjectPropertiesInSignature()
    
    bot_class = adapter.create_class(BOT)
    top_class = adapter.create_class(TOP)

    
    axioms = HashSet()
    for role in roles:
        for cls in classes:
            existential = adapter.create_object_some_values_from(role, cls)
            axiom1 = adapter.create_subclass_of(existential, top_class)
            axiom2 = adapter.create_subclass_of(bot_class, existential)
            axioms.add(axiom1)
            axioms.add(axiom2)

    print(f"Axioms before: {len(ontology.getAxioms())}")
    manager.addAxioms(ontology, axioms)
    print(f"Axioms after: {len(ontology.getAxioms())}")
            

                                                                                                                                                                                     


