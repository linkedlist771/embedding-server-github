import os

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_model_dir_path = os.path.join(root_dir, "embedding_models")
# _default_model_dir = "sentence-transformers_all-MiniLM-L6-v2"
_custom_model_list = [model_dir for model_dir in os.listdir(_model_dir_path) if os.path.isdir(os.path.join(_model_dir_path, model_dir))]
# model name should be specified in the get model function
# _custom_model_list.remove(_default_model_dir)
LOGGING_DIR = "logs"
EMBEDDING_MODEL_TYPES = ["default", "custom", ] #"ada"
COLLECTIONS_CONFIG = [
]
for custom_model in _custom_model_list:
    config = {
            "embedding_model_type": "custom",
            "embedding_model_path": os.path.join(root_dir, "embedding_models", custom_model),
            "model_index": _custom_model_list.index(custom_model),
     }
    COLLECTIONS_CONFIG.append(config)
SENTENCE_TRANSFORMERS_HOME = os.path.join(root_dir, "embedding_models")

if __name__ == "__main__":
    print(COLLECTIONS_CONFIG)

    # print(COLLECTIONS_CONFIG)

