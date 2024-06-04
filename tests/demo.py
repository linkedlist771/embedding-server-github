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