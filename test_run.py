from rag_pipeline import RAGPipeline, RAGConfig
print("🚀 Starting RAG Test...")
# Initialize
pipeline = RAGPipeline(RAGConfig())

# Load PDF
pipeline.load_documents(["sample.pdf"])   # make sure file exists

# Process (split + embeddings + FAISS)
pipeline.process_documents()

print("✅ Pipeline ready. You can now ask questions.")

# Ask question
while True:
    query = input("\nAsk question (or type exit): ")

    if query.lower() == "exit":
        break

    response = pipeline.query(query)

    if response["type"] == "relevant":
        print("\n✅ ANSWER FOUND:")
        for i, doc in enumerate(response["results"], 1):
            print(f"\nResult {i}:")
            print(doc["content"][:300])  # preview

    elif response["type"] == "fallback":
        print("\n⚠️ FALLBACK TRIGGERED")
        print("Keywords:", response["keywords"])
        print("Suggestions:", response["suggestions"])