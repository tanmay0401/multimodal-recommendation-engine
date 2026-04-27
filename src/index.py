import torch
import numpy as np
import faiss
import pandas as pd
from src.model import TwoTowerModel
import os

def build_index(embeddings_path="data/embeddings.npy", model_path="data/two_tower.pth", index_path="data/item_index.faiss"):
    print("Loading CLIP embeddings...")
    embeddings = np.load(embeddings_path)
    
    print("Loading Two-Tower model (Item Tower)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoTowerModel(clip_dim=512, latent_dim=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Projecting embeddings to latent space...")
    with torch.no_grad():
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        # Process in batches to avoid GPU OOM if necessary
        item_embs = []
        batch_size = 1024
        for i in range(0, len(embeddings_tensor), batch_size):
            batch = embeddings_tensor[i : i + batch_size]
            _, item_emb = model(batch, batch) # Using dummy user features for the forward call
            item_embs.append(item_emb.cpu().numpy())
            
        item_embs = np.concatenate(item_embs, axis=0)
    
    # FAISS IndexIVFFlat
    d = item_embs.shape[1]
    nlist = 100  # Number of clusters
    quantizer = faiss.IndexFlatIP(d) # Inner Product (for cosine similarity on normalized vectors)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print("Training FAISS index...")
    index.train(item_embs)
    
    print("Adding vectors to index...")
    index.add(item_embs)
    
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    print("Done!")

if __name__ == "__main__":
    if os.path.exists("data/embeddings.npy") and os.path.exists("data/two_tower.pth"):
        build_index()
    else:
        print("Required files (embeddings.npy or two_tower.pth) not found.")
