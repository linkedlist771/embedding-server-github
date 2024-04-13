from chromadb.utils import embedding_functions
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config
from sentence_transformers.models import Transformer
from collections import OrderedDict
import importlib
import json
import os


def load_jin_2_sentence_transformer(model_path):
    """
    Loads a full sentence-transformers model
    """

    def import_from_string(dotted_path):
        """
        Import a dotted module path and return the attribute/class designated by the
        last name in the path. Raise ImportError if the import failed.
        """
        try:
            module_path, class_name = dotted_path.rsplit(".", 1)
        except ValueError:
            msg = "%s doesn't look like a module path" % dotted_path
            raise ImportError(msg)

        try:
            module = importlib.import_module(dotted_path)
        except:
            module = importlib.import_module(module_path)

        try:
            return getattr(module, class_name)
        except AttributeError:
            msg = 'Module "%s" does not define a "%s" attribute/class' % (
                module_path,
                class_name,
            )
            raise ImportError(msg)

    def load(input_path: str):
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        # config['trust_remote_code'] = True
        print(f"config = {config}")
        return Transformer(model_name_or_path=input_path, **config)

    # Load the modules of sentence transformer
    modules_json_path = os.path.join(model_path, "modules.json")
    with open(modules_json_path) as fIn:
        modules_config = json.load(fIn)

    modules = OrderedDict()
    for module_config in modules_config:
        module_class = import_from_string(module_config["type"])
        module = load(os.path.join(model_path, module_config["path"]))
        modules[module_config["name"]] = module

    return modules

    pass


import numpy as np

jin_embedding_model_path = "../embedding_models/jina-embeddings-v2-base-en"
# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=jin_embedding_model_path)
from transformers import AutoModel
from numpy.linalg import norm

cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))
model = AutoModel.from_pretrained(jin_embedding_model_path, trust_remote_code=True)
# trust_remote_code is needed to use the encode method
embeddings = model.encode(["How is the weather today?"])
embedding0 = embeddings[0]
from sentence_transformers import SentenceTransformer

st = SentenceTransformer()  # model_name_or_path
# modules = st._load_sbert_model(jin_embedding_model_path)
modules = st._load_sbert_model(jin_embedding_model_path)
# st = SentenceTransformer(modules=modules)
st = SentenceTransformer(modules=[model])
# super().__init__(modules)
embedding = st.encode(["How is the weather today?"])
embedding = np.array(embedding)
embedding1 = embedding[0]
sim = cos_sim(embedding0, embedding1)
print(sim)
