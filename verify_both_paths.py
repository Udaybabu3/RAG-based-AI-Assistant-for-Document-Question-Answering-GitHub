"""
Verify BOTH paths of the RAG pipeline:
  1. Relevant query  -> returns document chunks
  2. Unrelated query  -> triggers fallback with links
"""
from rag_pipeline import RAGPipeline, RAGConfig

print("=" * 60)
print("  RAG Pipeline — Dual Path Verification")
print("=" * 60)

pipeline = RAGPipeline(RAGConfig(similarity_threshold=0.35))
pipeline.load_documents(["sample.pdf"])
pipeline.process_documents()
print("[OK] Pipeline initialized\n")

# ──────────────────────────────────────────────────────────
# TEST 1: Relevant query (matches resume content directly)
# ──────────────────────────────────────────────────────────
print("-" * 60)
print("TEST 1: RELEVANT QUERY")
print("Query: 'Peeka Uday Babu NIT Warangal education'")
print("-" * 60)

r1 = pipeline.query("Peeka Uday Babu NIT Warangal education")
print(f"  Response type   : {r1['type']}")
print(f"  Max similarity  : {r1['max_similarity']:.4f} ({r1['max_similarity']:.2%})")

if r1["type"] == "relevant":
    print(f"  Chunks returned : {len(r1['results'])}")
    for i, doc in enumerate(r1["results"], 1):
        cos = doc.get("cosine_similarity", 0)
        print(f"\n  --- Chunk {i} (cosine: {cos:.2%}) ---")
        print(f"  {doc['content'][:200]}...")
    print("\n  >> PASS: Document chunks returned correctly")
else:
    print("  >> FAIL: Expected 'relevant', got 'fallback'")

# ──────────────────────────────────────────────────────────
# TEST 2: Unrelated query (should trigger fallback)
# ──────────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("TEST 2: UNRELATED QUERY")
print("Query: 'How to cook pasta carbonara recipe'")
print("-" * 60)

r2 = pipeline.query("How to cook pasta carbonara recipe")
print(f"  Response type   : {r2['type']}")
print(f"  Max similarity  : {r2['max_similarity']:.4f} ({r2['max_similarity']:.2%})")

if r2["type"] == "fallback":
    print(f"  Keywords        : {r2.get('keywords', [])}")
    print(f"  Links generated :")
    for name, url in r2.get("links", {}).items():
        print(f"    - {name}: {url}")
    print("\n  >> PASS: Fallback triggered with search links")
else:
    print("  >> FAIL: Expected 'fallback', got 'relevant'")

# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
both_pass = r1["type"] == "relevant" and r2["type"] == "fallback"
if both_pass:
    print("  RESULT: BOTH PATHS VERIFIED SUCCESSFULLY")
else:
    print("  RESULT: ONE OR MORE PATHS FAILED")
print("=" * 60)
