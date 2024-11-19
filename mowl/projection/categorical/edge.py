import numpy as np
import torch as th
from org.semanticweb.owlapi.model import OWLClassExpression, OWLObjectProperty, OWLClass, OWLObjectUnionOf, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, ClassExpressionType, OWLObjectComplementOf



from org.semanticweb.owlapi.manchestersyntax.renderer import ManchesterOWLSyntaxOWLObjectRendererImpl
from jpype.types import JString
from org.mowl import MOWLShortFormProvider
from mowl.owlapi.defaults import BOT, TOP
from mowl.owlapi import OWLAPIAdapter
renderer = ManchesterOWLSyntaxOWLObjectRendererImpl()
short_form_provider = MOWLShortFormProvider()

renderer.setShortFormProvider(short_form_provider)

adapter = OWLAPIAdapter()
top_class = adapter.create_class(TOP)
bot_class = adapter.create_class(BOT)
import logging

class Node():
    def __init__(self, owl_class = None, relation = None, domain=False, codomain=False, negated_domain=False, is_individual=False):
        bad_class = False
        bad_relation = False
        if owl_class is not None and  not isinstance(owl_class, (OWLClassExpression)):
            bad_class = True

        if relation is not None and not isinstance(relation, OWLObjectProperty):
            bad_relation = True

        #Simplify the node
        ## Not Bot ---> Top and Not Top ---> Bot
        if isinstance(owl_class, OWLObjectComplementOf):
            owl_class_operand = owl_class.getOperand()
            if owl_class_operand.isOWLNothing():
                owl_class = top_class
            elif owl_class_operand.isOWLThing():
                owl_class = bot_class

        
                
        elif isinstance(owl_class, OWLObjectUnionOf):
            operands = owl_class.getOperandsAsList()
            if len(operands) == 1:
                raise ValueError(f"Union of one element is not allowed\n{owl_class}    \t{operands[0].toString()}")
            for op in operands:
                if op.isOWLNothing():
                    logging.info(f"owl:Nothing existing in union. Removing it..")
                    operands.remove(op)
                elif op.isOWLThing():
                    logging.info(f"owl:Thing existing in union. Simplifying to owl:Thing..")
                    operands = None
                    break
            if operands is None:
                owl_class = top_class
            elif len(operands) == 1:
                owl_class = operands[0]
            else:
                owl_class = adapter.create_object_union_of(*operands)

        
                
        if isinstance(owl_class, OWLObjectIntersectionOf):
            operands = owl_class.getOperandsAsList()
            if len(operands) == 1:
                raise ValueError(f"Intersection of one element is not allowed\n{owl_class}    \t{operands[0].toString()}")
            for op in operands:
                if op.isOWLThing():
                    logging.info(f"owl:Thing existing in intersection. Removing it..")
                    operands.remove(op)
                if op.isOWLNothing():
                    logging.info(f"owl:Nothing existing in intersection. Simplifying to owl:Nothing..")
                    operands = None
                    break

            if operands is None:
                owl_class = bot_class

            elif len(operands) == 1:
                owl_class = operands[0]
            else:
                owl_class = adapter.create_object_intersection_of(*operands)

                
        
        if bad_class or bad_relation:
            raise TypeError(f"Wrong either owl_class or relation. Required owl_clas of type OWLClassExpression. Got {type(owl_class)}. Required relation of type OWLObjectProperty. Got {type(relation)}") 

        if domain and codomain:
            raise ValueError("Domain and codomain cannot be both True")
        if negated_domain and not domain:
            raise ValueError("Negated domain can only be True if domain is True")

            
        if relation is not None and owl_class is not None:
            owl_class = owl_class.getNNF()
            
        if relation is not None and not owl_class is None and not relation.equals(owl_class.getProperty()):
            raise ValueError(f"Relation and owl_class do not match. Relation: {relation.toStringID()}. OWLClass: {owl_class.toStringID()}")


        

        self.domain = domain
        self.codomain = codomain
        self.negated_domain = negated_domain
        self.owl_class = owl_class
        self.relation = relation
        self.is_individual = is_individual

        if not owl_class is None and self.relation is None and not self.domain and not self.codomain:
            tmp_class = None
            if self.owl_class.getClassExpressionType() == ClassExpressionType.OBJECT_COMPLEMENT_OF:
                tmp_class = owl_class.getOperand()
                if tmp_class.getClassExpressionType() == ClassExpressionType.OBJECT_COMPLEMENT_OF:
                    self.owl_class = tmp_class.getOperand()
            

        
        assert not (domain and codomain)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False

        eq = True
        if self.owl_class is None:
            eq = eq and other.owl_class is None
        else:
            eq = eq and self.owl_class.equals(other.owl_class)

        if self.relation is None:
            eq = eq and other.relation is None
        else:
            eq = eq and self.relation.equals(other.relation)

        eq = eq and self.domain == other.domain
        eq = eq and self.codomain == other.codomain
        eq = eq and self.negated_domain == other.negated_domain

        return eq
            
    def __hash__(self):
        return hash((self.owl_class, self.relation, self.domain, self.codomain, self.negated_domain))

    def __repr__(self):
        if self.relation is None:
            expr_str = renderer.render(self.owl_class)
            expr_str = expr_str.replaceAll(JString("[\\r\\n|\\r|\\n()|<|>]"), JString(""))
            return str(expr_str)
        else:
            if self.owl_class is None:
                repr_ = str(self.relation.toStringID()) 
            else:
                cls_str = renderer.render(self.owl_class)
                cls_str = cls_str.replaceAll(JString("[\\r\\n|\\r|\\n()|<|>]"), JString(""))
                cls_str = str(cls_str)
                rel_str = str(self.relation.toStringID())
                repr_ = rel_str + "_under_" + cls_str

            if self.domain:
                repr_ = "DOMAIN_" + repr_
            elif self.codomain:
                repr_ = "CODOMAIN_" + repr_

            if self.negated_domain:
                repr_ = "NOT_" + repr_

            return repr_


    def is_negated(self):
        is_negated = False
        if self.owl_class is not None and self.relation is None:
            if self.owl_class.getClassExpressionType() == ClassExpressionType.OBJECT_COMPLEMENT_OF:
                is_negated = True
        elif self.domain:
            if self.negated_domain:
                is_negated = True

        return is_negated

    def is_intersection(self):
        is_intersection = False
        if self.owl_class is not None:
            if self.owl_class.getClassExpressionType() == ClassExpressionType.OBJECT_INTERSECTION_OF:
                is_intersection = True
        return is_intersection

    def is_union(self):
        is_union = False
        if self.owl_class is not None:
            if self.owl_class.getClassExpressionType() == ClassExpressionType.OBJECT_UNION_OF:
                is_union = True
        return is_union

        
    def negate(self):
        if not self.in_object_category():
            raise ValueError("Cannot negate a relation node")
        if self.domain:
            new_node = Node(owl_class=self.owl_class, relation=self.relation, domain=self.domain, codomain=self.codomain, negated_domain= not self.negated_domain)
        else:
            if self.owl_class.getClassExpressionType() == ClassExpressionType.OBJECT_COMPLEMENT_OF:
                new_node = Node(owl_class=self.owl_class.getOperand(), relation=self.relation, domain=self.domain, codomain=self.codomain, negated_domain=self.negated_domain)
            elif self.owl_class.isOWLThing():
                new_node = Node(owl_class=bot_class, relation=self.relation, domain=self.domain, codomain=self.codomain, negated_domain=self.negated_domain)
            elif self.owl_class.isOWLNothing():
                new_node = Node(owl_class=top_class, relation=self.relation, domain=self.domain, codomain=self.codomain, negated_domain=self.negated_domain)
            else:
                new_node = Node(owl_class=self.owl_class.getObjectComplementOf(), relation=self.relation, domain=self.domain, codomain=self.codomain, negated_domain=self.negated_domain)
        return new_node

    def nnf(self):
        if not self.in_object_category():
            return self
        if self.owl_class is None:
            return self
        else:
            new_node = Node(owl_class=self.owl_class.getNNF(), relation=self.relation, domain=self.domain, codomain=self.codomain, negated_domain=self.negated_domain)
            return new_node

    def to_domain(self):
        if not self.in_relation_category():
            raise ValueError("Cannot convert a class node to domain")
        if self.domain:
            raise ValueError("Cannot convert a domain node to domain")
        if self.codomain:
            raise ValueError("Cannot convert a codomain node to domain")
        if self.negated_domain:
            raise ValueError("Cannot convert a negated domain node to domain")

        new_node = Node(owl_class=self.owl_class, relation=self.relation, domain=True, codomain=False, negated_domain=False)
        return new_node

    def to_codomain(self):
        if not self.in_relation_category():
            raise ValueError("Cannot convert a class node to codomain")
        if self.domain:
            raise ValueError("Cannot convert a domain node to codomain")
        if self.codomain:
            raise ValueError("Cannot convert a codomain node to codomain")
        if self.negated_domain:
            raise ValueError("Cannot convert a negated domain node to codomain")

        new_node = Node(owl_class=self.owl_class, relation=self.relation, domain=False, codomain=True, negated_domain=False)
        return new_node



            
        
    def in_object_category(self):
        """Determines if the node is in the category of objects, not relations"""
        in_object_category = False

        if self.is_individual:
            return False
        
        if self.owl_class is not None:
            if self.relation is not None:
                if self.domain or self.codomain:
                    in_object_category = True
                else:
                    in_object_category = False
            else:
                in_object_category = True
        else:
            if self.domain or self.codomain:
                in_object_category = True
            else:
                in_object_category = False

        return in_object_category

    
    def in_relation_category(self):
        """Determines if the node is in the category of relations, not objects"""
        return not self.in_object_category()
        

    def is_whole_relation(self):
        if self.owl_class is None and self.relation is not None and not (self.domain or self.codomain or self.negated_domain):
            return True
        else:
            return False
    
    def is_existential(self):
        is_existential = False
        if self.in_object_category():
            if not self.domain and not self.codomain:
                if self.owl_class is not None and self.relation is not None:
                    if self.owl_class.getClassExpressionType() == ClassExpressionType.OBJECT_SOME_VALUES_FROM:
                        is_existential = True

        return is_existential

    def is_owl_nothing(self):
        is_bot = False
        if self.owl_class is not None:
            is_bot = self.owl_class.isOWLNothing()
        return is_bot

    def is_owl_thing(self):
        is_top = False
        if self.owl_class is not None:
            is_top = self.owl_class.isOWLThing()
        return is_top

    def get_operand(self):
        if not self.is_negated:
            raise ValueError("Node is not negated")
        if self.negated_domain:
            new_node = Node(owl_class=self.owl_class, relation=self.relation, domain=self.domain, codomain=self.codomain, negated_domain=False)
        elif self.owl_class.getClassExpressionType() == ClassExpressionType.OBJECT_COMPLEMENT_OF:
            new_node = Node(owl_class=self.owl_class.getOperand(), relation=self.relation, domain=self.domain, codomain=self.codomain, negated_domain=self.negated_domain)
        else:
            raise ValueError("Node is not negated")
        return new_node



    
class Edge:
    """Class representing a graph edge.
    """

    def __init__(self, src, rel, dst, weight=1.):
 
        if not isinstance(src, Node):
            raise TypeError("Parameter src must be a Node")
        if not isinstance(rel, str):
            raise TypeError("Parameter rel must be a str")
        if not isinstance(dst, Node):
            raise TypeError("Parameter dst must be a Node")
        if not isinstance(weight, float):
            raise TypeError("Optional parameter weight must be a float")

        src_str = str(src)
        dst_str = str(dst)

        self._src = src
        self._rel = rel
        self._dst = "" if dst == "" else dst
        self._weight = weight

    @property
    def src(self):
        """
        Getter method for ``_src`` attribute

        :rtype: str
        """
        return self._src

    @property
    def rel(self):
        """
        Getter method for ``_rel`` attribute

        :rtype: str
        """
        return self._rel

    @property
    def dst(self):
        """
        Getter method for ``_dst`` attribute

        :rtype: str
        """
        return self._dst

    @property
    def weight(self):
        """
        Getter method for ``_weight`` attribute

        :rtype: str
        """
        return self._weight

    def astuple(self):
        return tuple(map(str, (self._src, self._rel, self._dst)))

    @staticmethod
    def getEntitiesAndRelations(edges):
        return Edge.get_entities_and_relations(edges)

    @staticmethod
    def get_entities_and_relations(edges):
        '''
        :param edges: list of edges
        :type edges: :class:`Edge`

        :returns: Returns a 2-tuple containing the list of entities (heads and tails) and the \
            list of relations
        :rtype: (Set of str, Set of str)
        '''

        entities = set()
        relations = set()

        for edge in edges:
            entities |= {edge.src, edge.dst}
            relations |= {edge.rel}

        return (entities, relations)

    @staticmethod
    def zip(edges):
        return tuple(zip(*[x.astuple() for x in edges]))

    
