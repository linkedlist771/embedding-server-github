# Embedding Server

> Embedding server is used to provide embedding service based
> on sentence transformers models and is `fully aligned` with OpenAI's
> embedding service.
> Embedding has been deployed on GPU server
> remotely to provide embedding service.

埋め込みサーバーの詳細については、ご希望の言語を選択してください：
- [English](README.md)
- [日本語 (Japanese)](README-jp.md)


## 🆕 New: 
This project will be rebuild with `Rust` to provide a more efficient and faster service. 
Please check [embedding-server-github-rust](https://github.com/linkedlist771/embedding-server-github-rust) 
for details.

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
export HF_ENDPOINT=https://hf-mirror.com

git clone $HF_ENDPOINT/sentence-transformers/all-mpnet-base-v2
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
# Usage
- Using Openai's API to get embeddings:
```python
from openai import OpenAI

if __name__ == "__main__":
    # assume you start the server at localhost:8848
    api_base = "http://localhost:8848/v1"

    client = OpenAI(base_url=api_base)
    text = "Hello, world!"
    # you can change the model to the actual model you use
    model = "text2vec-base-chinese"
    res = client.embeddings.create(input=[text], model=model).data[0].embedding
    print(res)
```

- Curl
```bash
curl --location --request GET '127.0.0.1:8848/v1/get_collection_config' \
--header 'User-Agent: Apifox/1.0.0 (https://apifox.com)'
```



## API reference

https://apifox.com/apidoc/shared-fb1805a7-e3e7-4fce-9b9e-3bd69c45e171/api-123096464

# Contribution

PR is welcome, please follow the code style with [format.sh](format.sh).

## ToDo
- [ ] Support multiple instance to make work load balance.