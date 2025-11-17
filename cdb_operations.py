import chromadb

def main():
    client = chromadb.PersistentClient(path="data/chroma")
    collection_names = [collection.name for collection in client.list_collections()]
    print(collection_names)

    collection = client.get_collection("pandas_help_corpus_with_qwen3-embedding_0_6b")
    print(collection.metadata["embedding_model"])
    print(collection.configuration["hnsw"])

    # client.delete_collection(name="pandas_help_corpus")
    

if __name__ == "__main__":
    main()