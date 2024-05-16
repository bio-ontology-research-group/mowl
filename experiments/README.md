# Reproducibility instructions for experiments

```
mamba env create -f environment.yml
conda activate ontoem
cd ..
python setup.py install
```


# Data

You can get the data from [insert link here]

We used Gene Ontology at version 2024-04-24 and FoodOn at version 2024-04-07. You can easily download the original ontologies using:
```
python download_ontologies.py
```

| Ontology | All Axioms | All axioms C subclass D | Traninig | Validation | Testing | Training Deductive Closure | Traning Deductive Closure with no Top as superclass | Dreprecated Classes |
|----------|------------|-------------------------|----------|------------|---------|----------------------------|-----------------------------------------------------|---------------------|
| GO       | 95249      | 66810                   | 75207    | 6680       | 13362   | 276709                     | 222275                                              | 9216                |
| FoodOn   | 59385      | 45137                   | 45844    | 4513       | 9028    | 150481                     | 112849                                              | 2178                |





# Running

```
cd elembeddings
python subsumption.py -ns -e 100
```

We can use the same trained model to evaluate under different settings:


```
# To filter the deductive closure
python subsumption.py -ns -e 100 -ot -filterded
```


```
# To use the deductive closure as positives
python subsumption.py -ns -e 100 -evalded
```



# Preliminary Results


## ELEmbeddings - Gene Ontology

### Testing set only (ignoring deductive closure axioms)


| MR    | MRR    | AUC    | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 |
|-------|--------|--------|--------|--------|---------|---------|----------|
| 10561 | 0.0177 | 0.7948 | 0.0015 | 0.0183 | 0.0467  | 0.1119  | 0.1525   |


### Testing set including deductive closure axioms

| MR   | MRR    | AUC    | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 |
|------|--------|--------|--------|--------|---------|---------|----------|
| 8492 | 0.0307 | 0.8350 | 0.0052 | 0.0302 | 0.0786  | 0.1916  | 0.2550   |



