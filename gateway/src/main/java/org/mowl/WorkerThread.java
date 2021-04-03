package org.mowl;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.stream.Collectors;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class WorkerThread implements Runnable{


    public static void main(String[] args){}

    String out_file;
    HashMap<String, ArrayList<ArrayList<String>>> graph;
    int number_walks;
    int length_walk;
    String source;    


    public WorkerThread(String out_file, HashMap<String, ArrayList<ArrayList<String>>> graph, int number_walks, int length_walk, String source){
        
        this.out_file = out_file;
        this.graph = graph;
        this.number_walks = number_walks;
        this.length_walk = length_walk;
        this.source = source;
    }


    @Override
    public void run(){
        String to_print = "";

        HashMap<Long, ArrayList<String>> walks = new HashMap<Long, ArrayList<String>>();
        int source_size = this.graph.get(this.source).size();
        if (source_size > 0){ // if there are outgoing edges at all
            for (long i = 0; i < this.number_walks; i++){
                int count = this.length_walk;
                String current = this.source;
                
                ArrayList<String> walk = new ArrayList<String>();
                walk.add(this.source);
                walks.put(i, walk);
                while(count > 0){
                    int curr_node_length = this.graph.get(current).size();
                    if(curr_node_length > 0){ // # if there are outgoing edges
                        int random_val = getRandomNumber(0, curr_node_length);
                        ArrayList<String> neighbor = this.graph.get(current).get(random_val);
                        String edge = neighbor.get(0);
                        String target = neighbor.get(1);
                        walks.get(i).add(edge);
                        walks.get(i).add(target);
                        current = target;
                    }else{
                        String edge = "*******";
                        walks.get(i).add(edge);
                        walks.get(i).add(this.source);
                        current = this.source;
                    }
                    count -= 1;
                }
            }
        }

        for(long i = 0; i < walks.size(); i++){
            ArrayList<String> walk = walks.get(i);

            String walkString = walk.stream().collect(Collectors.joining(" "));

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

