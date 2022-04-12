from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import sys
import numpy as np 
import random
import math
import pickle
import tqdm


genes_vectors_filename = sys.argv[1]
diseases_vectors_filename = sys.argv[2]
positives_filename = sys.argv[3]
hard = False
ratio = 5
Vector_size = 100


# This method is used to generate negative samples 
# input:  
# # genes_keys: 
# # diseases_keys:
# # positives:
# # hard: 
# output:
# # negatives:
# # new_positives:
# # pos_count:
# # neg_count:
# 
# When data are generated the negative genes are selected in 2 ways: hard choise will select the negative genes from the disease associated genes only,
# not hard when the selection of the genes are from associated and non associated genes. 
def generate_negatives(genes_keys, diseases_keys, positives, hard):
	negatives = {}
	new_positives = {}
	pos_count = 0
	neg_count = 0
	disease_associated_genes = set([])
	for disease in positives:
		if (disease in diseases_keys):
			for gene in positives[disease]:
				if(gene in genes_keys):
					if(disease not in new_positives):
						new_positives[disease]=set([])
					pos_count+=1
					disease_associated_genes.add(gene)
					new_positives[disease].add(gene)
	non_disease_associated_genes = set([])
	for gene in genes_keys:
		if gene not in disease_associated_genes:
			non_disease_associated_genes.add(gene)

	#genes can be associated or non associated genes
	if not hard: 
		for disease in diseases_keys:
			if disease in positives:
				negatives[disease] = set([])
				for gene in genes_keys:
					neg_count+=1
					negatives[disease].add(gene)

	#genes are only the associated genes
	if hard:
		for disease in diseases_keys:
			if disease in positives:
				negatives[disease] = set([])
				for gene in genes_keys:
					if (gene not in positives[disease]) and gene not in non_disease_associated_genes:
						neg_count+=1
						negatives[disease].add(gene)
						break
	return negatives,new_positives, pos_count, neg_count

def get_input_analysis(genes_vectors_filename, diseases_vectors_filename, positives_filename):
	genes_vectors = {}
	with open(genes_vectors_filename,'r') as f:
		genes_vectors = json.load(f)

	diseases_vectors = {}
	with open(diseases_vectors_filename,'r') as f:
		diseases_vectors = json.load(f)

	positives = {}
	with open(positives_filename,'r') as f:
		positives = json.load(f)

	diseases_keys = list(diseases_vectors.keys())
	genes_keys = list(genes_vectors.keys())

	new_positives={}
	for disease in positives:
		if (disease in diseases_keys):
			for gene in positives[disease]:
				if(gene in genes_keys):
					if(disease not in new_positives):
						new_positives[disease]=set([])
					new_positives[disease].add(gene)

	new_disease_keys = [x for x in diseases_keys if x in new_positives]

	print(len(new_disease_keys), len(genes_keys) , len(new_positives.keys()))

	return new_disease_keys,genes_keys,new_positives


def get_input(genes_vectors_filename, diseases_vectors_filename ,positives_filename, ratio):
	genes_vectors = {}
	with open(genes_vectors_filename,'r') as f:
		genes_vectors = json.load(f)

	diseases_vectors = {}
	with open(diseases_vectors_filename,'r') as f:
		diseases_vectors = json.load(f)

	positives = {}
	with open(positives_filename,'r') as f:
		positives = json.load(f)

	diseases_keys = list(diseases_vectors.keys())
	genes_keys = list(genes_vectors.keys())

	negatives, new_positives, pos_count, neg_count = generate_negatives(genes_keys, diseases_keys, positives, hard)


	# Defining Feature Matrex
	X= np.empty(((ratio+1)*pos_count,Vector_size*2))
	y= np.empty((ratio+1)*pos_count)

	negative_diseases = list(negatives.keys())
	sample_number=0
	for disease in new_positives:
		for gene in new_positives[disease]:
			x = np.concatenate((diseases_vectors[disease],genes_vectors[gene]),axis=0)
			X[sample_number]=x
			y[sample_number]=1
			sample_number+=1


			for i in range(ratio):
				n = random.randint(0,len(negative_diseases))
				n_disease = negative_diseases[n-1]
				n = random.randint(0,len(negatives[n_disease]))
				n_gene = list(negatives[n_disease])[n-1]
				x = np.concatenate((diseases_vectors[n_disease],genes_vectors[n_gene]),axis=0)
				X[sample_number]=x
				y[sample_number]=0
				sample_number+=1
	return X,y

def get_training_folds(genes_vectors_filename, diseases_vectors_filename ,positives,diseases_keys,genes_keys, ratio, fold):
	genes_vectors = {}
	with open(genes_vectors_filename,'r') as f:
		genes_vectors = json.load(f)

	diseases_vectors = {}
	with open(diseases_vectors_filename,'r') as f:
		diseases_vectors = json.load(f)

	start = int(len(diseases_keys)*fold/10)
	end = int(len(diseases_keys)*(fold+1)/10) - 1


	testing_disease_keys = diseases_keys[start:end]
	training_and_validation = [x for x in diseases_keys if x not in testing_disease_keys]
	split = int(len(training_and_validation)*0.8)
	training_disease_keys = training_and_validation[:split]
	validation_disease_keys = training_and_validation[split+1:]

	#print(start,end,len(testing_disease_keys),len(training_disease_keys)

	negatives_t, new_positives_t, pos_count_t, neg_count_t = generate_negatives(genes_keys, training_disease_keys, positives, hard)
	negatives_v, new_positives_v, pos_count_v, neg_count_v = generate_negatives(genes_keys, validation_disease_keys, positives, hard)


	# Defining folds dic (will be saved as pickle files)
	training_dic = {}
	validation_dic = {}
	testing_dic = {}
	negative_diseases = list(negatives_t.keys())


	for disease in new_positives_t:
		for gene in new_positives_t[disease]:
			training_dic[(disease,gene)]={}
			training_dic[(disease,gene)]['d']=diseases_vectors[disease]
			training_dic[(disease,gene)]['g']=genes_vectors[gene]
			training_dic[(disease,gene)]['label']=1

			for i in range(ratio):
				n = random.randint(1,len(negative_diseases))
				n_disease = negative_diseases[n-1]
				n = random.randint(1,len(negatives_t[n_disease]))
				n_gene = list(negatives_t[n_disease])[n-1]
				training_dic[(n_disease,n_gene)]={}
				training_dic[(n_disease,n_gene)]['d']=diseases_vectors[n_disease]
				training_dic[(n_disease,n_gene)]['g']=genes_vectors[n_gene]
				training_dic[(n_disease,n_gene)]['label']=0


	negative_diseases = list(negatives_v.keys())
	for disease in new_positives_v:
		for gene in new_positives_v[disease]:
			validation_dic[(disease,gene)]={}
			validation_dic[(disease,gene)]['d']=diseases_vectors[disease]
			validation_dic[(disease,gene)]['g']=genes_vectors[gene]
			validation_dic[(disease,gene)]['label']=1

			for i in range(ratio):
				n = random.randint(1,len(negative_diseases))
				n_disease = negative_diseases[n-1]
				n = random.randint(1,len(negatives_v[n_disease]))
				n_gene = list(negatives_v[n_disease])[n-1]
				validation_dic[(n_disease,n_gene)]={}
				validation_dic[(n_disease,n_gene)]['d']=diseases_vectors[n_disease]
				validation_dic[(n_disease,n_gene)]['g']=genes_vectors[n_gene]
				validation_dic[(n_disease,n_gene)]['label']=0


	X_test= np.empty((len(testing_disease_keys)*len(genes_keys),Vector_size*2))
	y_test= np.empty(len(testing_disease_keys)*len(genes_keys))
	for disease in testing_disease_keys:
		for gene in genes_keys:
			testing_dic[(disease,gene)]={}
			testing_dic[(disease,gene)]['d']=diseases_vectors[disease]
			testing_dic[(disease,gene)]['g']=genes_vectors[gene]
			if(disease in positives):
				if(gene in positives[disease]):
					testing_dic[(disease,gene)]['label']=1
				else:
					testing_dic[(disease,gene)]['label']=0
			else:
				testing_dic[(disease,gene)]=0

	with open('training_fold'+str(fold)+'.pickle', 'wb') as handle:
		pickle.dump(training_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('validation_fold'+str(fold)+'.pickle', 'wb') as handle:
		pickle.dump(validation_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('testing_fold'+str(fold)+'.pickle', 'wb') as handle:
		pickle.dump(testing_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return training_dic,validation_dic,testing_dic



if (True):
	HDs_keys,OGs_keys,positives = get_input_analysis(genes_vectors_filename, diseases_vectors_filename, positives_filename)
	OGs_HDs_sim = np.empty((len(HDs_keys),len(OGs_keys)))

	for fold in range(10):
		print("-------------statring fold--------------")
		print(fold)
		train, validation, testing = get_training_folds(genes_vectors_filename, diseases_vectors_filename, positives,HDs_keys, OGs_keys, ratio, fold)
