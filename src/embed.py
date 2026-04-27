import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os
from tqdm import tqdm
import argparse

def generate_embeddings(input_path="data/products.parquet", output_path="data/embeddings.npy", limit=None, batch_size=128):
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    if limit:
        print(f"Limiting to {limit} items for dry run.")
        df = df.head(limit)
    
    print(f"Loading CLIP model (clip-ViT-B-32)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SentenceTransformer('clip-ViT-B-32', device=device)
    
    # Combine title and description for semantic embedding
    texts = (df['title'] + " " + df['description']).tolist()
    
    print(f"Generating embeddings for {len(texts)} items...")
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print(f"Saving embeddings to {output_path} (Shape: {embeddings.shape})...")
    np.save(output_path, embeddings)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items for testing")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding generation")
    args = parser.parse_args()
    
    generate_embeddings(limit=args.limit, batch_size=args.batch_size)
