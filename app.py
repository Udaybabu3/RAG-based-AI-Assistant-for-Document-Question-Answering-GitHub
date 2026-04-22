"""
RAG-based AI Assistant for Document Question Answering
Streamlit UI with out-of-scope detection and configurable similarity threshold.
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import RAGPipeline, RAGConfig

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────────────────────
# Custom Styling
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ─────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0;
        opacity: 0.85;
        font-size: 1rem;
    }

    /* ── Result cards ───────────────────────────────────── */
    .result-card {
        background: #f8f9fc;
        border-left: 4px solid #667eea;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.65;
    }
    .result-card .score-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* ── Fallback box ───────────────────────────────────── */
    .fallback-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 0.5rem;
    }
    .fallback-box a {
        color: #e65100;
        font-weight: 600;
        text-decoration: none;
    }
    .fallback-box a:hover {
        text-decoration: underline;
    }

    /* ── Sidebar ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #f4f4f8;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.1rem;
        color: #333;
    }

    /* ── Status badges ──────────────────────────────────── */
    .status-ready {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-waiting {
        background: #fff3e0;
        color: #e65100;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 RAG Document Q&A</h1>
    <p>Upload documents, ask questions — with smart out-of-scope detection</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Sidebar: Configuration
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help=(
            "Minimum cosine similarity to accept a result as relevant. "
            "Queries scoring below this trigger the fallback system. "
            "Lower -> more permissive · Higher -> stricter. "
            "Recommended: 0.20-0.25 for sparse docs, 0.30-0.40 for dense docs."
        ),
    )

    top_k = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of document chunks to retrieve per query.",
    )

    chunk_size = st.slider(
        "Chunk Size (chars)",
        min_value=200,
        max_value=2000,
        value=500,
        step=100,
        help="Size of each text chunk during document splitting.",
    )

    chunk_overlap = st.slider(
        "Chunk Overlap (chars)",
        min_value=0,
        max_value=500,
        value=100,
        step=50,
        help="Overlap between consecutive chunks to preserve context.",
    )

    st.divider()
    st.markdown("## 📊 Pipeline Status")

    if "pipeline" in st.session_state and st.session_state.pipeline is not None:
        st.markdown('<span class="status-ready">✅ Pipeline Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-waiting">⏳ Awaiting Documents</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Document Upload
# ─────────────────────────────────────────────────────────
col_upload, col_status = st.columns([3, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to build the knowledge base.",
    )

with col_status:
    if uploaded_files:
        st.metric("Files Uploaded", len(uploaded_files))

# ─────────────────────────────────────────────────────────
# Pipeline Initialization
# ─────────────────────────────────────────────────────────
if uploaded_files:
    # Rebuild pipeline when files or config change
    config_key = (similarity_threshold, top_k, chunk_size, chunk_overlap)
    file_key = tuple(f.name for f in uploaded_files)
    cache_key = (file_key, config_key)

    if st.session_state.get("cache_key") != cache_key:
        with st.spinner("🔄 Processing documents — building vector index…"):
            config = RAGConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                similarity_threshold=similarity_threshold,
                top_k=top_k,
            )
            pipeline = RAGPipeline(config)

            # Save uploaded files to temp paths
            temp_paths = []
            for uf in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(uf.read())
                tmp.close()
                temp_paths.append(tmp.name)

            pipeline.load_documents(temp_paths)
            pipeline.process_documents()

            # Clean up temp files
            for p in temp_paths:
                os.unlink(p)

            st.session_state.pipeline = pipeline
            st.session_state.cache_key = cache_key

        st.success(f"✅ Indexed **{len(uploaded_files)}** document(s) successfully!")

# ─────────────────────────────────────────────────────────
# Query Interface
# ─────────────────────────────────────────────────────────
st.markdown("---")
query = st.text_input(
    "💬 Ask a question about your documents",
    placeholder="e.g. What are the key findings in section 3?",
)

if query and "pipeline" in st.session_state and st.session_state.pipeline is not None:
    pipeline: RAGPipeline = st.session_state.pipeline

    # Update threshold in case user changed slider after init
    pipeline.config.similarity_threshold = similarity_threshold
    pipeline.config.top_k = top_k

    with st.spinner("🧠 Searching documents…"):
        response = pipeline.query(query)

    # ── Display: Relevant results ──────────────────────────
    if response["type"] == "relevant":
        max_sim = response.get("max_similarity", 0)
        st.markdown(
            f"### ✅ Relevant Results Found  \n"
            f"**Best match similarity:** `{max_sim:.2%}`"
        )

        for i, doc in enumerate(response["results"], 1):
            cos = doc.get("cosine_similarity", 0)
            st.markdown(
                f"""<div class="result-card">
                    <span class="score-badge">#{i} — Similarity: {cos:.2%}</span>
                    <p>{doc["content"]}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Display: Fallback ──────────────────────────────────
    elif response["type"] == "fallback":
        max_sim = response.get("max_similarity", 0)
        st.markdown(
            f"### ⚠️ Out-of-Scope Query  \n"
            f"**Best match similarity:** `{max_sim:.2%}` — below threshold `{similarity_threshold:.2%}`"
        )

        links = response.get("links", {})
        links_html = "".join(
            f'<li><strong>{name}:</strong> <a href="{url}" target="_blank">{url}</a></li>'
            for name, url in links.items()
        )

        st.markdown(
            f"""<div class="fallback-box">
                <p>🔍 <strong>This question is outside the scope of the uploaded documents.</strong></p>
                <p>You can explore it here:</p>
                <ul>{links_html}</ul>
            </div>""",
            unsafe_allow_html=True,
        )

        # Also show extracted keywords
        keywords = response.get("keywords", [])
        if keywords:
            st.markdown(f"**Extracted Keywords:** {', '.join(keywords)}")

elif query and ("pipeline" not in st.session_state or st.session_state.pipeline is None):
    st.warning("⬆️ Please upload a PDF document first.")

# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.82rem;'>"
    "Built with 🤗 HuggingFace Embeddings · FAISS · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
