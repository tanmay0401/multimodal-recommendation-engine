# ROADMAP.md

> **Current Phase**: Not started
> **Milestone**: v1.0

## Must-Haves (from SPEC)
- [ ] CLIP embedding generation for 100k items.
- [ ] Trained Two-Tower retrieval model.
- [ ] FAISS index integration.
- [ ] LightGBM ranking model.
- [ ] Functional FastAPI endpoint.

## Phases

### Phase 1: Data & Embedding Foundation
**Status**: ⬜ Not Started
**Objective**: Generate synthetic dataset and compute joint image-text embeddings using CLIP.
**Requirements**: REQ-01, REQ-02

### Phase 2: Retrieval Engine (Two-Tower)
**Status**: ⬜ Not Started
**Objective**: Build and train the PyTorch Two-Tower architecture for latent space mapping.
**Requirements**: REQ-03, REQ-04

### Phase 3: Fast Search & Indexing
**Status**: ⬜ Not Started
**Objective**: Index item vectors in FAISS and implement efficient top-K retrieval.
**Requirements**: REQ-05

### Phase 4: Precision Ranking
**Status**: ⬜ Not Started
**Objective**: Train the LightGBM model for re-ranking candidates and implement the simulation for CTR improvement.
**Requirements**: REQ-06, REQ-09

### Phase 5: API & Deployment
**Status**: ⬜ Not Started
**Objective**: Serve the full pipeline via FastAPI and implement the image upload feature.
**Requirements**: REQ-07, REQ-08
