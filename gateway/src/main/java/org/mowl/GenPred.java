package org.mowl;

import java.util.*;
import java.util.stream.*;
import java.lang.Math;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;


public class GenPred implements Runnable{

    public static void main(String[] args){}        
    
    String word1;
    ArrayList<String> vocab;
    ArrayList<String> relations;
    HashMap<String, ArrayList<Float>> dict_vocab;
    ArrayList<ArrayList<String>> preds;

    public GenPred(String word1, HashMap<String, ArrayList<Float>> dict_vocab, ArrayList<String> relations, ArrayList<ArrayList<String>> preds){
        this.word1 = word1;
        this.vocab = new ArrayList<>(dict_vocab.keySet());
        this.dict_vocab = dict_vocab;
	this.relations = relations;
        this.preds = preds;
    }
        

    @Override
    public void run(){
        if(contains(word1, "http://4932.")){
            for(int j=0; j<vocab.size(); j++){
                String word2 = vocab.get(j);
                if (!word1.equals(word2) && contains(word2, "http://4932.")){
                    
                    float distance = similarity(dict_vocab.get(word1), dict_vocab.get(word2));
                    ArrayList<String> triple = new ArrayList<String>(){{
                        add(word1);
                        add(word2);
                        add(String.valueOf(distance));
                    }};
                    synchronized(lock){
                        preds.add(triple);
                        try{
                            writeToFile(word1+ ", " + word2 + ", " + String.valueOf(distance) + "\n", "data/Predictions.txt");
                        }catch(IOException e) {
                            e.printStackTrace();
                        }
                
                    }
                }
            }
        }        
    }

    private static Object lock = new Object(); // the object for locks does not have to be anything special.


    public static void writeToFile(String to_print, String out_file) throws IOException{

        // File f = new File("out.gz");
        // OutputStream os = new GZIPOutputStream(new FileOutputStream(f, true)); // true for append
        // PrintWriter w = new PrintWriter(new OutputStreamWriter(os));
        // w.println("log message");


        BufferedWriter writer = new BufferedWriter(new FileWriter(out_file, true));
        writer.append(to_print);
        writer.close();   
    }



    public float similarity(ArrayList<Float> a, ArrayList<Float> b){
        List<Integer> seq = IntStream.range(0, a.size()).boxed().collect(Collectors.toList());  

        float dot = seq.stream()
                        .reduce((float) 0, (acc, idx) -> acc + a.get(idx)*b.get(idx), Float::sum);//) a.get(idx)*b.get(idx));

        float normA = (float) Math.sqrt(a.stream()
                            .reduce((float) 0, (acc, item) -> acc + item*item)
                            );
        float normB = (float) Math.sqrt(b.stream()
                            .reduce((float) 0, (acc, item) -> acc + item*item)
                            );
        return dot/(normA*normB);
    }

    public boolean contains(String superString, String substring){
        return superString.indexOf(substring) != -1;
    }

}
