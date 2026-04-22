from rag_pipeline import RAGPipeline, RAGConfig
print("🚀 Starting RAG Test...")
# Initialize
pipeline = RAGPipeline(RAGConfig())

# Load PDF
pipeline.load_documents(["sample.pdf"])   # make sure file exists

# Process (split + embeddings + FAISS)
pipeline.process_documents()

print("✅ Pipeline ready. You can now ask questions.")
print(f"ℹ️  Similarity threshold: {pipeline.config.similarity_threshold}")

# Ask question
while True:
    query = input("\nAsk question (or type exit): ")

    if query.lower() == "exit":
        break

    response = pipeline.query(query)

    if response["type"] == "relevant":
        print(f"\n✅ ANSWER FOUND  (best similarity: {response['max_similarity']:.2%})")
        for i, doc in enumerate(response["results"], 1):
            cos = doc.get("cosine_similarity", 0)
            print(f"\nResult {i} (cosine: {cos:.2%}):")
            print(doc["content"][:300])  # preview

    elif response["type"] == "fallback":
        print(f"\n⚠️ FALLBACK TRIGGERED  (best similarity: {response['max_similarity']:.2%})")
        print(response["message"])
        print("Keywords:", response.get("keywords", []))