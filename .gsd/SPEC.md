# SPEC.md — Project Specification

> **Status**: `FINALIZED`

## Vision
A production-grade Multi-Modal Semantic Recommendation System demonstrating a two-stage retrieval and ranking pipeline for 100k+ products using joint image-text embeddings.

## Goals
1. **Multi-Modal Embeddings**: Leverage OpenAI CLIP to create unified representations of products from images and text.
2. **Efficient Retrieval**: Implement a PyTorch Two-Tower model trained with contrastive loss for latent space mapping.
3. **Scalable Search**: Use FAISS (IndexIVFFlat/HNSW) to achieve sub-millisecond top-K retrieval across 100k items.
4. **Precision Re-Ranking**: Train a LightGBM model to optimize CTR based on similarity, user history, and product features.
5. **Real-time API**: Serve recommendations via a FastAPI endpoint supporting both user ID and image-based queries.

## Non-Goals (Out of Scope)
- Real-time user history updates (batch training/updates are sufficient for this version).
- Full front-end application (API-only for this phase).
- Complex authentication/authorization logic.

## Users
Developers and Hiring Managers looking for a demonstration of high-scale ML engineering and multi-modal search.

## Constraints
- **Data**: Synthetic 100k item catalog (Faker) or Amazon Product Reviews.
- **Tech Stack**: Python, PyTorch, FAISS, CLIP, LightGBM, FastAPI.
- **Performance**: Sub-100ms end-to-end inference latency.

## Success Criteria
- [ ] Successful training of Two-Tower model with descending contrastive loss.
- [ ] Retrieval latency < 10ms for 100k items using FAISS.
- [ ] NDCG@10 and Recall@K reported on a held-out test set.
- [ ] Working FastAPI endpoint returning ranked lists.
