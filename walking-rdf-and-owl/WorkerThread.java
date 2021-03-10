//package com.worker;


import java.util.HashMap;
import java.util.ArrayList;
import java.util.stream.Collectors;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class WorkerThread implements Runnable{


    public static void main(String[] args){}

    String out_file;
    HashMap<Integer, ArrayList<ArrayList<Integer>>> graph;
    int number_walks;
    int length_walk;
    Integer source;    


    public WorkerThread(String out_file, HashMap<Integer, ArrayList<ArrayList<Integer>>> graph, int number_walks, int length_walk, Integer source){
        this.out_file = out_file;
        this.graph = graph;
        this.number_walks = number_walks;
        this.length_walk = length_walk;
        this.source = source;

    }


    @Override
    public void run(){
        String to_print = "";

        HashMap<Integer, ArrayList<Integer>> walks = new HashMap<Integer, ArrayList<Integer>>();
        
        int source_size = this.graph.get(this.source).size();
        if (source_size > 0){ // if there are outgoing edges at all
            for (int i = 0; i < this.number_walks; i++){
                int count = this.length_walk;
                int current = this.source;
                
                ArrayList<Integer> walk = new ArrayList<Integer>();
                walk.add(this.source);
                walks.put(i, walk);
                while(count > 0){
                    int curr_node_length = this.graph.get(current).size();
                    if(curr_node_length > 0){ // # if there are outgoing edges
                        int random_val = getRandomNumber(0, curr_node_length);
                        ArrayList<Integer> neighbor = this.graph.get(current).get(random_val);
                        Integer edge = neighbor.get(0);
                        Integer target = neighbor.get(1);
                        walks.get(i).add(edge);
                        walks.get(i).add(target);
                        current = target;
                    }else{
                        Integer edge = Integer.MAX_VALUE;
                        walks.get(i).add(edge);
                        walks.get(i).add(this.source);
                    }
                    count -= 1;
                }
            }
        }

        for(int i = 0; i < walks.size(); i++){
            ArrayList<Integer> walk = walks.get(i);

            String walkString = walk.stream().map(Object::toString).collect(Collectors.joining(" "));

            to_print += walkString + '\n';
        }

        try{
            writeToFile(to_print, this.out_file);
        }catch(IOException e) {
            e.printStackTrace();
        }

    }


    private static Object lock = new Object(); // the object for locks does not have to be anything special.

    public static void writeToFile(String to_print, String out_file) throws IOException{
        BufferedWriter writer = new BufferedWriter(new FileWriter(out_file, true));
        synchronized (lock){
            writer.append(to_print);
            writer.close();    
        }
    }
    

    
    public int getRandomNumber(int min, int max){
        return (int) ((Math.random() * (max - min)) + min);
    }
}

