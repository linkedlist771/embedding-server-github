# Embedding Server

> Embedding server is used to provide embedding service based
> on sentence transformers models and is fully aligned with OpenAI's
> embedding service.
> Embedding has been deployed on GPU server
> remotely to provide embedding service.

## Installation
```bash
pip install -r requirements.txt
```

## Download models
> The embedding server is based on sentence transformers models.
> Here are the recommended models:
> - [all-mpnet-base-v2(Best but slower.)](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
> - [multi-qa-MiniLM-L6-cos-v1(Medium)](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)
> - [all-MiniLM-L6-v2(Fast but not that good.)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Take `all-mpnet-base-v2` as an example:

```bash
mkdir embedding_models
git lfs install
git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
mv all-mpnet-base-v2 embedding_models/
```

## Deployment

- Run server on GPU sever:

```bash
CUDA_VISIBLE_DEVICES=0 python embedding_server.py 
--models_dir_path embedding_deployment_model/ 
--use_gpu --port port --host host

```


for example:
```bash
CUDA_VISIBLE_DEVICES=0 python embedding_server.py 
--models_dir_path embedding_deployment_model/ 
--use_gpu --port 8000 --host 0.0.0.0

```

## Usage

https://apifox.com/apidoc/shared-fb1805a7-e3e7-4fce-9b9e-3bd69c45e171/api-123096464