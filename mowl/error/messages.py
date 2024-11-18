"""Module that containts names of typical errors in mOWL"""
OWLAPI_DIRECT = "For direct access to OWLAPI use Java imports."
INVALID_WALKER_NAME = "Invalid walker name. Valid names are: deepwalk, node2vec."
EMBEDDINGS_NOT_FOUND_MODEL_NOT_TRAINED = "Embeddings not found. Model has not been trained yet."

GRAPH_MODEL_PROJECTOR_NOT_SET = "An instance of GraphModel needs a projector. Please add one using the set_projector method."

RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND = "RandomWalkModel does not contain embeddings. Model has not been trained yet."

RANDOM_WALK_MODEL_WALKER_NOT_SET = "RandomWalkModel does not have a walker. Please add one using the set_walker method."

W2V_MODEL_NOT_SET = "This model requires a Word2Vec model. Please add one using the set_w2v_model method."

KGE_METHOD_NOT_SET = "This model requires a KGE method. Please add one using the 'set_kge_method' method."

CORPUS_NOT_GENERATED = "Corpus not generated. Please generate it using the 'generate_corpus' method."

W2V_FROM_PRETRAINED_MODEL_ALREADY_SET = "Word2Vec model already set. Please set overwrite to True if you want to overwrite it."

PYKEEN_FROM_PRETRAINED_MODEL_ALREADY_SET = "PyKEEN model already set. Please set overwrite to True if you want to overwrite it."

MODEL_ALREADY_SET = "Model already set. Please set overwrite to True if you want to overwrite it."

PYKEEN_OPTIMIZER_NOT_SET = "PyKEEN optimizer not set. Please set it by doing 'model.optimizer = your_optimizer'."

PYKEEN_LR_NOT_SET = "PyKEEN learning rate not set. Please set it by doing 'model.lr = your_learning_rate'."

PYKEEN_BATCH_SIZE_NOT_SET = "PyKEEN batch size not set. Please set it by doing 'model.batch_size = your_batch_size'."

PYKEEN_MODEL_NOT_SET = "PyKEEN model not set. Please set it by using the 'set_kge_method' method."

EVALUATOR_NOT_SET = "Evaluator not set. Please set it using the 'set_evaluator' method."

MODEL_NOT_TRAINED_OR_LOADED = "Model has not been trained or loaded yet. Use 'model.train' or 'model.from_pretrained'."


def type_error(parameter_name, expected_type, parameter_type, optional=False):
    if optional:
        prefix = "Optional parameter"
    else:
        prefix = "Parameter"
    
    return f"{prefix} {parameter_name} must be of type {expected_type}. Got {parameter_type} instead."
