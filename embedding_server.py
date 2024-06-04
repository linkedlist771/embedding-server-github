import os
from contextlib import asynccontextmanager
from fastapi.responses import Response
from fastapi import APIRouter
from starlette.exceptions import HTTPException as StarletteHTTPException
import argparse
import fire
from robyn import Robyn, Request
from robyn import SubRouter, jsonify, WebSocket
from robyn import logger
import json
import copy
from schema import (
    Usage,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
)

from utils.embedding_utils import (
    get_model_list,
    load_model,
    load_tokenizer,
    get_embedding,
    get_logger,
    get_current_cuda_device,
)

from config.embedding_config import COLLECTIONS_CONFIG


parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0", help="host")
parser.add_argument("--port", default=8000, help="port")
parser.add_argument(
    "--models_dir_path", required=True, help="path to the models directory"
)
parser.add_argument("--use_gpu", action="store_true", help="flag to use GPU")
parser.add_argument(
    "--tokenizer_model", default="cl100k_base", help="tokenizer model name"
)
args = parser.parse_args()



def on_start():
    global supported_models, tokenizer
    logger.info("Starting to load models...")
    if args.use_gpu:
        logger.info(
            f"GPU is enabled. The current model will be loaded on cuda: {get_current_cuda_device()}."
        )
    else:
        logger.info("GPU is disabled. The current model will be loaded on CPU.")
        # 我觉得时间都花在传输哪个embedding tensor上了。 
    model_infos = get_model_list(args.models_dir_path)
    try:
        for model_info in model_infos:
            model_name = model_info["label"]
            model_path = os.path.join(args.models_dir_path, model_name)
            models[model_name] = load_model(model_path, use_gpu=args.use_gpu)
            supported_models += f"{model_name}, "
            logger.info(f"Model {model_name} loaded successfully.")
        tokenizer = load_tokenizer(args.tokenizer_model)
        logger.info(f"Tokenizer {args.tokenizer_model} loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


# app = FastAPI(docs_url=None, redoc_url=None)
models = {}
tokenizer = None
supported_models: str = ""
# get the logger
# logger = get_logger(__name__)
router = SubRouter(__name__, prefix="/v1")


# init the collection
collections_config = copy.deepcopy(COLLECTIONS_CONFIG)
for collection_config in collections_config:
    # (COLLECTIONS_CONFIG, collection_type, emb_model_type="default", custom_index=0)
    emb_model_type = collection_config["embedding_model_type"]
    custom_index = (
        collection_config.get("model_index", 0) if emb_model_type == "custom" else 0
    )
logger.info("Collections initialized successfully.")



# Embedding related.




app = Robyn(__file__)

async def startup_handler():
    on_start()


@app.shutdown_handler
def shutdown_handler():
    global models, tokenizer
    models.clear()

@app.post("/")
async def h(request: Request):
    return request.body

@router.get("/get_model_types")
async def model_type() -> list:
    logger.info("Received request for model types.")
    model_types = get_model_list(args.models_dir_path)
    logger.info("Sending response for model types.")
    return model_types


@router.get("/get_collection_config")
async def collection_config() -> list:
    logger.info("Received request for collection config.")
    logger.info("Sending response for collection config.")
    return collections_config
from viztracer import log_sparse


@router.post("/embeddings")
@log_sparse
async def embeddings(request: Request) -> EmbeddingResponse:
    embedding_request = request.body
    logger.info(f"embedding_request: {embedding_request}")
    embedding_request = json.loads(embedding_request)
    embedding_request = EmbeddingRequest.model_validate(embedding_request)
    logger.info(f"Received embedding request for model: {embedding_request.model}")

    model_name = embedding_request.model
    # if model_name not in models:
    #     logger.error(f"Only support models: {supported_models}, {model_name} not found")
    #     raise HTTPException(
    #         status_code=404,
    #         detail=f"Only support models: {supported_models}, {model_name} not found",
    #     )
    model = models.get(model_name)

    # if not model:
    #     logger.error(f"Model {model_name} not found")
    #     raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    input_text = embedding_request.input
    embeddings_list = get_embedding(model, input_text, use_gpu=args.use_gpu)
    # Simulate the usage information, you need to replace this with actual values
    prompt_tokens = (
        len(tokenizer.encode(input_text))
        if isinstance(input_text, str)
        else len(tokenizer.encode("".join(input_text)))
    )
    usage_info = Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)
    # Construct the list of Embedding objects
    data = [
        Embedding(object="embedding", index=i, embedding=emb)
        for i, emb in enumerate(embeddings_list)
    ]
    # Create the EmbeddingResponse object
    embedding_response = EmbeddingResponse(
        object="list", data=data, model=model_name, usage=usage_info
    ).model_dump()

    logger.info("Embedding response sent.")
    return jsonify(embedding_response)


def start_server(port=args.port, host=args.host):
    logger.info(f"Starting server at {host}:{port}")
    app.include_router(router)
    app.startup_handler(startup_handler)

    try:
        app.start(port=port, host=host)
    finally:
        logger.info("Server shutdown.")


if __name__ == "__main__":
    fire.Fire(start_server)
