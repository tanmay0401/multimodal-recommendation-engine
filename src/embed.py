import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os
from tqdm import tqdm
import argparse
from PIL import Image

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
    
    # Load images
    print("Loading local images for embedding...")
    images = []
    valid_indices = []
    
    for idx, path in enumerate(tqdm(df['image_url'])):
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            # Skip images that fail to load
            pass
            
    # If some failed, filter them out from the dataframe
    if len(valid_indices) < len(df):
        print(f"Warning: {len(df) - len(valid_indices)} images failed to load. Filtering dataset.")
        df = df.iloc[valid_indices].reset_index(drop=True)
        df.to_parquet(input_path, index=False)
        
    print(f"Generating embeddings for {len(images)} images...")
    embeddings = model.encode(
        images, 
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
