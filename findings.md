RAG Evaluation Framework — Key Learnings

1\. Library Version Conflicts



RAGAS 0.2.6 incompatible with langchain-google-genai 2.0.5

Root cause: newer langchain-google-genai changed how it passes temperature to Gemini API

Fix: downgraded to langchain-google-genai 1.0.10



2\. Silent Metric Failures (nan scores)



All RAGAS metrics returned nan without raising exceptions

Root causes: LLM judge not explicitly initialised + reference fields not grounded in actual retrieved contexts

Fix: explicitly passed llm= and embeddings= to evaluate(), rewrote references to match vault content only



3\. API Quota Management



Bottleneck was request count (RPD/RPM), not token usage — RAGAS fires dozens of small calls per eval run

Exhausted both gemini-2.0-flash and gemini-2.0-flash-lite free tier quotas in one day

Fix: switched models, added rate limiting, made regrounding step optional to halve quota consumption



4\. Production Thinking



Identified that eval pipelines must account for API rate limits by design, not as an afterthought

Learned to monitor RPM/RPD separately from TPM — they hit different ceilings



5\. Empirical Evaluation & Performance Metrics

We executed a comprehensive evaluation comparing our baseline **Cosine Vector Retriever** with an ensemble **Hybrid Retriever** (0.5 BM25 + 0.5 Cosine) across our personal Obsidian knowledge base vault. 

### Ingestion & Vault Stats
- **Total Markdown Files Ingested:** 76 files
- **Total Indexed Document Chunks:** 178 chunks (chunk size: 512, overlap: 50)
- **Semantic Cache Optimization:** Configured at a cosine similarity threshold of `0.92`, which successfully eliminated duplicate external LLM requests.

### Mean Performance Scores (RAGAS Evaluation)

| RAGAS Metric | Cosine Retriever | Hybrid Retriever (BM25 + Cosine) | Metric Delta (Relative) |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | **0.9167** | 0.6786 | -0.2381 (-25.97%) |
| **Answer Relevancy** | 0.5941 | **0.6186** | +0.0245 (+4.12%) |
| **Context Recall** | **0.4583** | **0.4583** | 0.0000 (0.00%) |
| **Factual Correctness** | 0.4233 | **0.5767** | +0.1534 (**+36.24%**) |

### Performance Insights
- **Factual Correctness Boost:** The Hybrid Retriever achieved a stellar **+36.24% relative improvement** in Factual Correctness (`0.5767` vs `0.4233`). This demonstrates that combining keyword matching (BM25) with semantic embeddings is highly effective at retrieving exact terms (e.g., project names, specific tool titles) from unstructured vault files.
- **Answer Relevancy:** Relevancy slightly improved with the Hybrid retriever (`0.6186` vs `0.5941`), showing a more aligned final response.
- **The Faithfulness Trade-Off:** While Hybrid retriever retrieved more factual source context, it suffered from a **decline in Faithfulness** (`0.6786` vs `0.9167`). This indicates that hybrid chunk blends can introduce minor context noise or competing information that confuses the LLM generation step, leading to hallucinations or ungrounded assertions compared to the cleaner, purely semantic cosine retrieval.
- **Identical Context Recall:** Both retrievers achieved a context recall of `0.4583`. This indicates that for the highly structured, specific query set, both methods are retrieving a similar proportion of the ground truth documents, though the hybrid method retrieves a more factually descriptive subset.
