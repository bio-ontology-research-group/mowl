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

| Ontology | All Axioms | All axioms C subclass D | Traninig | Validation | Testing | Training Deductive Closure | Traning Deductive Closure with no Top as superclass |
|----------|------------|-------------------------|----------|------------|---------|----------------------------|-----------------------------------------------------|
| GO       | 95249      | 66810                   | 75206    | 6681       | 13362   | 314557                     | 216319                                              |
| FoodOn   | 59385      | 45137                   | 45844    | 4513       | 9028    | 194031                     | 126348                                              |


