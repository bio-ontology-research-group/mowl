"""Module that containts names of typical errors in mOWL"""
OWLAPI_DIRECT = "For direct access to OWLAPI use Java imports."
INVALID_WALKER_NAME = "Invalid walker name. Valid names are: deepwalk, node2vec."
EMBEDDINGS_NOT_FOUND_MODEL_NOT_TRAINED = "Embeddings not found. Model has not been trained yet."

GRAPH_MODEL_PROJECTOR_NOT_SET = "An instance of GraphModel needs a projector. Please add one using the set_projector method."

RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND = "RandomWalkModel does not contain embeddings. Model has not been trained yet."

RANDOM_WALK_MODEL_WALKER_NOT_SET = "RandomWalkModel does not have a walker. Please add one using the set_walker method."

W2V_MODEL_NOT_SET = "This model requires a Word2Vec model. Please add one using the set_w2v_model method."
