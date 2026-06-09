"""Real-benchmark instantiation of RelationalRAG-Tables (Phase 1+).

Separate from the dependency-free synthetic artifact (``relational_rag_tables``):
this package loads public benchmarks (HybridQA, then BIRD/Spider), builds a dense
retriever (sentence-transformers + FAISS), trains the learned link verifier on real
gold links, and runs an MLX-hosted LLM planner. Requires the heavy stack in
``requirements-real.txt`` on python3.13.
"""
