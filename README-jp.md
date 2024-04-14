# 埋め込みサーバー

> 埋め込みサーバーは、センテンストランスフォーマーモデルに基づく埋め込みサービスを提供し、OpenAIの埋め込みサービスと`完全に一致`しています。
> GPUサーバーにリモートでデプロイされた埋め込みは、埋め込みサービスを提供します。

埋め込みサーバーの詳細については、ご希望の言語を選択してください：
- [English](README.md)
- [日本語 (Japanese)](README-jp.md)


## 🆕 新着情報:
このプロジェクトは、より効率的で高速なサービスを提供するために`Rust`で再構築されます。
詳細は[embedding-server-github-rust](https://github.com/linkedlist771/embedding-server-github-rust)をご覧ください。

## インストール
```bash
pip install -r requirements.txt
```

## モデルのダウンロード
> 埋め込みサーバーはセンテンストランスフォーマーモデルに基づいています。
> 推奨されるモデルは以下の通りです:
> - [all-mpnet-base-v2（最高だが遅い。）](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
> - [multi-qa-MiniLM-L6-cos-v1（中程度）](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)
> - [all-MiniLM-L6-v2（速いがそれほど良くない。）](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

`all-mpnet-base-v2`を例に取ります:

```bash
mkdir embedding_models
git lfs install
export HF_ENDPOINT=https://hf-mirror.com

git clone $HF_ENDPOINT/sentence-transformers/all-mpnet-base-v2
mv all-mpnet-base-v2 embedding_models/
```

## デプロイメント

- GPUサーバーでサーバーを実行:

```bash
CUDA_VISIBLE_DEVICES=0 python embedding_server.py 
--models_dir_path embedding_deployment_model/ 
--use_gpu --port port --host host

```

例えば:
```bash
CUDA_VISIBLE_DEVICES=0 python embedding_server.py 
--models_dir_path embedding_deployment_model/ 
--use_gpu --port 8000 --host 0.0.0.0

```
# 使用方法
- OpenaiのAPIを使用して埋め込みを取得:
```python
from openai import OpenAI

if __name__ == "__main__":
    # サーバーをlocalhost:8848で開始すると仮定
    api_base = "http://localhost:8848/v1"

    client = OpenAI(base_url=api_base)
    text = "こんにちは、世界！"
    # 実際に使用するモデルに変更できます
    model = "text2vec-base-chinese"
    res = client.embeddings.create(input=[text], model=model).data[0].embedding
    print(res)
```

- Curl
```bash
curl --location --request GET '127.0.0.1:8848/v1/get_collection_config' \
--header 'User-Agent: Apifox/1.0.0 (https://apifox.com)'
```

## API リファレンス

https://apifox.com/apidoc/shared-fb1805a7-e3e7-4fce-9b9e-3bd69c45e171/api-123096464

# 寄稿

PRは歓迎されています。コードスタイルは[format.sh](format.sh)に従ってください。

## ToDo
- [ ] 複数インスタンスのサポートを行い、負荷分散を実現します。