import torch
from chromadb.utils import embedding_functions
from typing import Union
import logging
import os
from config.embedding_config import LOGGING_DIR
from logging.handlers import TimedRotatingFileHandler


def patch_sentence_transformer_load():
    from sentence_transformers.models import Transformer
    from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config

    def _load_model(self, model_name_or_path, config, cache_dir):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir)
        else:
            # need to add trust remote code here.
            self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir,
                                                        trust_remote_code=True)

    # monkey patching
    Transformer._load_model = _load_model


def get_current_cuda_device():
    return torch.cuda.current_device()


def get_model_path_list(model_list_dir):
    dir_and_file = os.listdir(model_list_dir)
    model_list = [dir for dir in dir_and_file if os.path.isdir(os.path.join(model_list_dir, dir))]
    return model_list


def get_model_list(model_list_Dir):
    model_path_list = get_model_path_list(model_list_Dir)
    return [{"label": model, "value": index + 1} for index, model in enumerate(model_path_list)]


def get_embedding(model, input_text: Union[list, str], use_gpu=False):
    _input_text = None
    if isinstance(input_text, list):
        _input_text = input_text
    elif isinstance(input_text, str):
        _input_text = [input_text]
    else:
        raise ValueError("Invalid input text type.")
    if use_gpu:
        return model._model.encode(  # type: ignore
            _input_text,
            convert_to_numpy=True,
            normalize_embeddings=model._normalize_embeddings,
            device=torch.device(torch.cuda.current_device()),
        ).tolist()
    else:
        return model(_input_text)


def load_model(model_path, use_gpu=False):
    device = "cuda" if use_gpu else "cpu"
    if use_gpu:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_path, device=device)
        embedding_function._model = embedding_function._model.to(device)
        embedding_function._target_device = torch.device(device)
    else:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_path, device=device)
    return embedding_function


def get_logger(name, level=logging.INFO, use_stream_handler=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    if use_stream_handler:
        logger.addHandler(sh)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logging_dir = os.path.join(current_dir, LOGGING_DIR)
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    file_handler = TimedRotatingFileHandler(
        f"{logging_dir}/{name}",
        when="midnight",
        interval=1,
        backupCount=1000,
    )

    file_handler.suffix += ".log"
    # Trigger a rollover immediately
    file_handler.doRollover()
    file_handler.close()
    # Remove the original log file
    os.remove(f"{logging_dir}/{name}")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# def main():
#     logger1 = get_logger("test_logger1")
#     logger1.info("hello world")
#
#     logger2 = get_logger("test_logger2")
#     logger2.info("hello world")
#
#     try:
#         raise Exception("test")
#     except Exception as e:
#         logger1.error("error", exc_info=e)
#         logger2.error(e)
#     finally:
#         pass
#
#
# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    MODEL_LIST_DIR = "../scripts/sentence-transformers/sts/embedding_deployment_model"
    model_list = get_model_path_list(MODEL_LIST_DIR)
    model_path = os.path.join(MODEL_LIST_DIR, model_list[0])
    model = load_model(model_path, use_gpu=False)
    sentence = "hello world"
    vector = get_embedding(model, sentence)
    print(f"type of vector: {type(vector)}")
    import numpy as np
    vector_np = np.array(vector)
    print(f"shape of vector: {vector_np.shape}")
    sentence = ["hello world"]*100
    times = 10
    import time
    cpu_start = time.time()
    # try a list of the same sentence
    for _ in range(times):
        vector = get_embedding(model, sentence)
    cpu_end = time.time()
    print(f"cpu time: {cpu_end-cpu_start}")

    model = load_model(model_path, use_gpu=True)
    gpu_start = time.time()
    for _ in range(times):
        vector = get_embedding(model, sentence, use_gpu=True)
    gpu_end = time.time()
    print(f"gpu time: {gpu_end-gpu_start}")

    print(f"type of vector: {type(vector)}")
    vector_np = np.array(vector)
    print(f"shape of vector: {vector_np.shape}")
    while True:
        vector = get_embedding(model, sentence)
        vector = np.array(vector)
        print(vector.shape)