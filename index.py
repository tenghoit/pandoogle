import json
from pathlib import Path
import chromadb
import ollama
import sys
import numpy as np
from datetime import datetime
import time

def generate_log_file() -> Path:
    time = datetime.now().replace(microsecond=0).strftime("%Y%m%d%H%M%S")
    log_path = Path(f"logs/{time}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path

def log(log_path: Path, text: str):
    time = datetime.now().replace(microsecond=0)
    full_text = f"[{time}] {text}"
    with open(log_path, "a") as f:
        f.write(full_text + "\n")
        print(text)

def normalize(v):
    return (v / np.linalg.norm(v)).tolist()


def normalize_collection_name(collection_name: str) -> str:
    invalid_chars = [":", "."]
    for char in invalid_chars:
        collection_name = collection_name.replace(char, "_")
    return collection_name


def get_or_create_collection(input_path: Path = Path("data/pandas_help_corpus.json"), model: str = "qwen3-embedding:8b") -> chromadb.Collection:
    
    client = chromadb.PersistentClient(path="data/chroma")
    name = input_path.stem
    collection_name = f"{name}_with_{model}"
    collection_name = normalize_collection_name(collection_name)

    if collection_name in [collection.name for collection in client.list_collections()]:
        print(f"Collection found: {collection_name}")
        return client.get_collection(name=collection_name)
    

    log_path = generate_log_file()
    # create
    
    log(log_path, f"Creating collection: {collection_name}")
    collection = client.create_collection(
        name=collection_name, 
        metadata={"embedding_model": model},
        configuration={"hnsw": {"space": "cosine"}}
    )

    log(log_path, f"Loading JSON")
    with open(input_path, "r") as f:
        chunks = json.load(f)
    log(log_path, f"JSON loaded")

    for i, chunk in enumerate(chunks):
        text = f"{chunk["symbol"]} {chunk["signature"]}\n{chunk["doc"]}"
        response = ollama.embed(model=model, input=text)
        embedding = np.array(response["embeddings"][0])
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk["doc"]],
            metadatas=[{
                "symbol": chunk["symbol"],
                "signature": chunk["signature"]
            }]
        )
        log(log_path, f"Processed {i+1}/{len(chunks)} chunks")

    log(log_path,f"Collection created")

    return collection
    

def query_collection(collection: chromadb.Collection, query: str, n_results: int = 5) -> chromadb.QueryResult:
    embedding_model = collection.metadata["embedding_model"]
    embedding_start_time = time.time()
    query_embedding = ollama.embed(model=embedding_model, input=query)["embeddings"][0]
    embedding_end_time = time.time()
    print(f"Embedding duration: {embedding_end_time-embedding_start_time}")
    query_embedding = np.array(query_embedding)

    search_start_time = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    search_end_time = time.time()
    print(f"Search duration: {search_end_time-search_start_time}")
    # return clean_results(results)
    return results

def clean_results(results):
    ids = results["ids"][0]
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # flattens results
    outputs = []
    for i in range(len(ids)):
        output = {
            "id": ids[i],
            "symbol": metadatas[i]["symbol"],
            "signature": metadatas[i]["signature"],
            "metadatas": metadatas[i],
            "doc": docs[i],
            "distance": distances[i]
        }
        outputs.append(output)

    return outputs


def main():
    current_model = "qwen3-embedding:0.6b"
    if len(sys.argv) > 1:
        current_model = sys.argv[1]

    input_path = Path("data/pandas_help_corpus.json")
    collection = get_or_create_collection(input_path, current_model)

    # query = f"prefix labels"
    # results = query_collection(collection, query)
    # print(type(results), len(results))
    # print(f"Dists: {results["distances"][0]}")


if __name__ == "__main__":
    main()