package org.mowl;

import java.util.*;
import java.util.stream.Collectors;
import java.io.IOException;

import com.google.common.collect.*;

public class WROEval implements Runnable{


    public static void main(String[] args){}

    ArrayList<String> pair;
    int k;
    ArrayList<ArrayList<String>> groundTruth;
    ArrayList<ArrayList<String>> predictions;
    ArrayList<String> subjects;
    HashMap<String, Integer> dict_subj_hits;
    HashMap<String, HashMap<Float, Integer>> dict_subj_ranks;
    


    public WROEval(ArrayList<String> pair, int k, ArrayList<ArrayList<String>> groundTruth, ArrayList<ArrayList<String>> predictions, ArrayList<String> subjects, HashMap<String, Integer> dict_subj_hits, HashMap<String, HashMap<Float, Integer>> dict_subj_ranks
    ){
        this.pair = pair;
        this.k = k;
        this.groundTruth = groundTruth;
        this.predictions = predictions;
        this.subjects = subjects;
        this.dict_subj_hits = dict_subj_hits;
        this.dict_subj_ranks = dict_subj_ranks;
    
    }


    private class Pair implements Comparable<Pair>{
        String subj;
        String obj;
        float score;

        public Pair(String subj, String obj, float score){
            this.subj = subj;
            this.obj = obj;
            this.score = score;
        }

        public Pair(String subj, String obj, String score){
            this.subj = subj;
            this.obj = obj;
            this.score = Float.parseFloat(score);
        }

        public Pair(ArrayList<String> data){
            this.subj = data.get(0);
            this.obj = data.get(1);
            this.score = Float.parseFloat(data.get(2));
        }

        public String toString(){
            return "(" + subj + ", " + obj + ", " + String.valueOf(score) + ")"; 
        }

        @Override
        public int compareTo(Pair other) {
            int subjects = this.subj.compareTo(other.subj);
            int objects = this.obj.compareTo(other.obj);
            return subjects * objects;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
    
            if (o == null || getClass() != o.getClass()) return false;
    
            Pair other = (Pair) o;
    
            return subj.equals(other.subj) && obj.equals(other.obj);
        }
    }

    @Override
    public void run(){
        String subj = pair.get(0);
        if (dict_subj_hits.containsKey(subj)){ //if already processed
            return;
        }

        //Extract triplets with fixed subj
        Set<Pair> grouped_pairs_gt = groundTruth.stream()
                                                .filter(x -> x.get(0) == subj)
                                                .map(x -> new Pair(x))
                                                .collect(Collectors.toSet());

        Set<Pair> grouped_pairs_pred = predictions.stream()
                                                .filter(x -> x.get(0) == subj)
                                                .map(x -> new Pair(x.get(0), x.get(1), x.get(2)))
                                                .collect(Collectors.toSet());

        // Set<Pair> grouped_pairs_pred_aux = grouped_pairs_pred.stream()
        //                                         .map(x -> new Pair(x.subj, x.obj, 0))
        //                                         .collect(Collectors.toSet());

        HashSet<Pair> all_pairs = new HashSet<>();
        for(int i=0; i<subjects.size(); i++){
            all_pairs.add(new Pair(subj, subjects.get(i), 0));
        }
        all_pairs.removeAll(grouped_pairs_pred);
        all_pairs.addAll(grouped_pairs_pred);


        List<Pair> grouped_pairs_gt_list = new ArrayList<Pair>(grouped_pairs_gt);
        List<Pair> all_pairs_list = new ArrayList<Pair>(all_pairs);

        List<Float> scores = all_pairs.stream()
        .map(x -> -x.score).
        collect(Collectors.toList());

        float[] ranking = rankify(scores, scores.size());
        // for (int i = 0; i < scores.size(); i++)
        //     System.out.print(ranking[i] + "  ");
        int hits = 0;
        HashMap<Float, Integer> ranks = new HashMap<>();
        
        for(int i=0; i<grouped_pairs_gt.size(); i++){
            Pair grouped_pair = grouped_pairs_gt_list.get(i);
            
            int idx = all_pairs_list.indexOf(grouped_pair);
            System.out.println("Index is " + idx + "\t\tScore is " + scores.get(idx) + "\t\tRank is " + ranking[idx]);
               
            float rank = ranking[idx];
            if(rank <= k){
                hits++;
            }
            if(! ranks.containsKey(rank))
                ranks.put(rank, 0);
            ranks.put(rank, ranks.get(rank) + 1);
        }
        
        synchronized(lock){
            dict_subj_hits.put(subj,hits);
            dict_subj_ranks.put(subj,ranks);

        }
    }


    private static Object lock = new Object(); // the object for locks does not have to be anything special.


    public List<Integer> rank_simple(List<Float> vector, int n){
        List<Integer> toOrder = new ArrayList<>();
        for (int i = 0; i<n; i++){
            toOrder.add( i);
        }

        // Collections.sort(toOrder, Ordering.explicit(vector).onResultOf(item -> item.getId()));
        
        // ArrayList<Integer> toReturn = new ArrayList<>();
        // for (int i = 0; i<n; i++){
        //     toReturn.add(Math.round(toOrder.get(i)));
        // }
        // return toReturn;

        toOrder.sort(Comparator.comparingInt(vector::indexOf));
        return toOrder;

        //return sorted(range(len(vector)), key=vector.__getitem__)
    }
    public float[] rankify(List<Float> A, int n){
        // Rank Vector     
        
        //float R[] = new float[n];
        
        
        // Sweep through all elements in A
        // for each element count the number
        // of less than and equal elements
        // separately in r and s
        // for (int i = 0; i < n; i++) {
        //     int r = 1, s = 1;
             
        //     for (int j = 0; j < n; j++)
        //     {
        //         if (j != i && A.get(j) < A.get(i))
        //             r += 1;
                     
        //         if (j != i && A.get(j) == A.get(i))
        //             s += 1;    
        //     }
         
        // Use formula to obtain  rank
        // R[i] = r + (float)(s - 1) / (float) 2;
        // }

        List<Integer> ivec = rank_simple(A, n);
        List<Float> svec = ivec.stream()
                    .map(rank -> A.get(rank))
                    .collect(Collectors.toList());
        int sumranks = 0;
        int dupcount = 0;
        float[] newarray = new float[n];
        for(int i=0; i<n; i++){
            sumranks = sumranks + i;
            dupcount++;
            if(i==n-1 || svec.get(i) != svec.get(i+1)){
                float averank = sumranks / (float) dupcount + 1;
                for(int j=i-dupcount+1; j<i+1; j++){
                    newarray[ivec.get(j)] = averank;
                }
                sumranks = 0;
                dupcount = 0;
            }
        }
        return newarray;
         
    }


}

