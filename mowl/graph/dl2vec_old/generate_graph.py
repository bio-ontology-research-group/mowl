import networkx as nx
import pickle as pkl
import json
import sys
import dgl
from mowl.graph.dl2vec.tree import Stack
from mowl.graph.edge import Edge

from org.mowl.DL2Vec import AxiomsFromOnt



def split_sentence(sentence):
    new_sentence=""
    for d in sentence:
        d = str(d)
        if d=="(":
            new_sentence+="( "
        elif d==")":
            new_sentence+=" )"
        else:
            new_sentence+=d
    return new_sentence

def test_stack(sentence):
    sentence = split_sentence(sentence)
    sentence =sentence.split(" ")
    stack = Stack()
    for data in sentence:
        if data=="":
            pass
        else:
            stack.push(data)
    print(stack.items)

    while stack.is_empty()== False:

        print(stack.pop())

def judge_or_and_inside(data):

    if ("and" in data) or ("or" in data):
        return True
    else:
        return False

def judge_restrictions(data):

    restrctions = ["exactly","min","max","some","only"]
    for d in data:
        if d in restrctions:
            return True
    return False

def judge_condition(stack):
    if stack.is_empty()==False:
        if stack.peak() !="(":

            return True
        else:
            return False


    else:
        return False

def convert_triple(data):
    restrictions = ["some","only","exactly","min","max"]

    result = []
    first_entity = data[0]
    stack = Stack()

    tag=False
    for entity in data[2:]:
        if entity ==")":
            tag=True
            temp_data =[]
            while stack.peak() !="(":
                temp_data.append(stack.pop())
            stack.pop()
            if judge_restrictions(temp_data):
                new_relation = temp_data[-1]
                tail_node = temp_data[0]
                new_node = new_relation+" "+tail_node
                stack.push(new_node)
            else:
                if (not stack.is_empty()):
                    if stack.peak() =="(" or stack.peak() =="and" or stack.peak()=="or":


                        for da in temp_data:
                            stack.push(da)


                    else:
                        new_temp_data=[]
                        while judge_condition(stack):
                            new_temp_data.append(stack.pop())
                        if new_temp_data!=[]:
                            new_relation = new_temp_data[-1]
                            for da in temp_data:
                                if da !="and" and da !="or":
                                    new_element = new_relation+" "+da
                                    stack.push(new_element)
                            # for da in new_temp_data:
                            #     if da !="and" and da !="or":
                            #         new_element = new_relation+" "+da
                            #         stack.push(new_element)
                        else:

                            for da in temp_data:
                                stack.push(da)

        else:
            stack.push(entity)

    if tag:
        final_axioms = []
        while(stack.is_empty()==False):
            final_axioms.append(stack.pop())

        for element in final_axioms:
            if (element in restrictions):
                end_node = final_axioms[0]
                new_relation = final_axioms[-1]
                axiom = new_relation+" "+end_node
                final_axioms=[axiom]
                break

        for axiom in final_axioms:
            if axiom!="and" and axiom !="or":
                axiom=first_entity+" "+axiom


                axiom=axiom.split(" ")
                result.append(axiom)


        return result
    else:
        final_axioms = []
        while(stack.is_empty()==False):
            axiom =stack.pop()
            if axiom !="and" and axiom!="or":
                final_axioms.append(axiom)


        end_node = final_axioms[0]
        new_relation=final_axioms[-1]
        axiom = new_relation+" "+end_node
        axiom=first_entity+" "+axiom


        axiom=axiom.split(" ")

        result.append(axiom)


        return result


def convert_graph(data):
    sentence = split_sentence(data)
    sentence = sentence.split(" ")
    if len(sentence) <3:
        pass
    elif len(sentence)==3:
        result =[[sentence[0], sentence[1], sentence[2]]]
        new_result=[]
        for da in result:
            new_result.append([da[0]," ".join(da[1:-1]),da[-1]])

        return new_result
    else:

        result = convert_triple(sentence)
        new_result=[]
        for da in result:
            new_result.append([da[0]," ".join(da[1:-1]),da[-1]])


        return new_result


def parseOWL(ontology):

    parser = AxiomsFromOnt(ontology, "elk")

    edges = parser.processOntology()
    
    # the restriction are min,max,exactly,some,only

    # there are conjunction or disjunction
    axiom_orig = parser.getAxioms()

    edges = []
    
    for line in axiom_orig:
        result = convert_graph(str(line).strip())

        # print("-"*40)
        for entities in result:
            new_edge = Edge(entities[0].strip(), entities[1].strip(), entities[2].strip())
            edges.append(new_edge)


    rels = {e.rel() for e in edges}
    #print(rels)
    return edges



