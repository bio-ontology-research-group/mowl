from pathlib import Path
from jpype import *
import jpype.imports
import os
import click as ck
from random import randrange
import numpy as np
import sys
import multiprocessing

from concurrent import futures

from time import sleep

# Obtain jar files.
jars_dir = "../gateway/build/distributions/gateway/lib/"
jars = f'{str.join(":", [jars_dir+name for name in os.listdir(jars_dir)])}'
jars += ':worker.jar' 

#Start JVM
startJVM(getDefaultJVMPath(), "-ea",  "-Djava.class.path=" + jars,  convertStrings=False)


from java.util.concurrent import ExecutorService  
from java.util.concurrent import Executors  
from java.lang import Runnable
from java.util import HashMap, ArrayList
import java.util.Map

@ck.command()
@ck.option(
    '--input', '-i', default= 'test_data/edgelist.txt',
    help='input edge list file ')
@ck.option(
    '--output', '-o', default= 'test_data/walks.txt',
    help='output file with DeepWalk output')
@ck.option(
    '--number-walks', '-nw', default=100,
    help='number of walks per node')
@ck.option(
    '--length-walk', '-lw', default=20,
    help='length of each walk')


def main(input, output, number_walks, length_walk):

    print("Building graph from " + input + "\n")
    graph = build_graph(input)
    print("Number of nodes in graph: " + str(len(graph)) + "\n")
    print("Writing walks to " + output + "\n")
    generate_corpus(output, graph, number_walks, length_walk)



def build_graph(filepath):
    edge_list = HashMap()
    
    with open(filepath, 'r') as f:
        count = 1
        for line in f:

            # TODO: optimize the following line, casting takes too much time ().
            node1, node2, edge = tuple(map (lambda x: (jpype.JObject(int(x), JClass("java.lang.Integer"))), line.rstrip('\n').split('\t')))

            neighbor = ArrayList()
            neighbor.add(edge)
            neighbor.add(node2)

            
            if edge_list.containsKey(node1):
                edge_list.get(node1).add(neighbor)
     
            else:
                neighbors = ArrayList()
                neighbors.add(neighbor)
                edge_list.put(node1, ArrayList(neighbors))
            count += 1
    return edge_list


def generate_corpus(out_file, graph, number_walks, length_walk):
    
    n_cores = multiprocessing.cpu_count()

    executor =  Executors.newFixedThreadPool(n_cores)

    #Just to clear the file before writing again on it.
    f = open(out_file, 'w')
    f.close()

    sources = ArrayList(graph.keySet())
    
    out_file_j = jpype.JObject(out_file, JClass("java.lang.String"))

    with jpype.synchronized(graph):
        for i in range(len(sources)):
            worker = JClass("WorkerThread")(out_file_j, graph, number_walks, length_walk, sources[i])
            executor.execute(worker)
            
        executor.shutdown()

if __name__ == '__main__':
    main()
