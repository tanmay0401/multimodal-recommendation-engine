# REQUIREMENTS.md

## Format
| ID | Requirement | Source | Status |
|----|-------------|--------|--------|
| REQ-01 | Generate 100k+ synthetic product items with image URLs and metadata. | SPEC Goal 1 | Pending |
| REQ-02 | Generate CLIP embeddings for all product items. | SPEC Goal 1 | Pending |
| REQ-03 | Implement PyTorch Two-Tower model (User Tower + Item Tower). | SPEC Goal 2 | Pending |
| REQ-04 | Train model using contrastive loss (InfoNCE) with in-batch negatives. | SPEC Goal 2 | Pending |
| REQ-05 | Build FAISS index for high-speed top-K item retrieval. | SPEC Goal 3 | Pending |
| REQ-06 | Implement LightGBM re-ranking model using similarity and user features. | SPEC Goal 4 | Pending |
| REQ-07 | Create FastAPI endpoint `POST /recommend`. | SPEC Goal 5 | Pending |
| REQ-08 | Implement logic for image-based recommendation steering. | SPEC Goal 5 | Pending |
| REQ-09 | Report evaluation metrics (Recall@K, NDCG, latency). | Success Criteria | Pending |
