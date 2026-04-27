import torch
import numpy as np
import faiss
import pandas as pd
from src.model import TwoTowerModel
import time

class Recommender:
    def __init__(self, model_path="data/two_tower.pth", index_path="data/item_index.faiss", products_path="data/products.parquet"):
        print("Initializing Recommender...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Model
        self.model = TwoTowerModel(clip_dim=512, latent_dim=128)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load Index
        self.index = faiss.read_index(index_path)
        
        # Load Metadata
        self.products = pd.read_parquet(products_path)
        
    def recommend(self, user_feature_clip, k=10):
        """
        user_feature_clip: [512] CLIP embedding representing user profile
        """
        start_time = time.time()
        
        with torch.no_grad():
            user_feat_tensor = torch.tensor(user_feature_clip, dtype=torch.float32).unsqueeze(0).to(self.device)
            user_emb, _ = self.model(user_feat_tensor, user_feat_tensor) # Dummy item features
            user_emb = user_emb.cpu().numpy().astype('float32')
            
        # FAISS search
        scores, indices = self.index.search(user_emb, k)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        results = self.products.iloc[indices[0]].copy()
        results['relevance_score'] = scores[0]
        
        return results, latency_ms

if __name__ == "__main__":
    # Example usage (will only work after Phase 3 is executed)
    import os
    if os.path.exists("data/item_index.faiss"):
        recommender = Recommender()
        
        # Mock a user feature (random for testing)
        mock_user_feat = np.random.randn(512).astype('float32')
        mock_user_feat /= np.linalg.norm(mock_user_feat)
        
        results, latency = recommender.recommend(mock_user_feat)
        
        print(f"\nTop Recommendations (Latency: {latency:.2f}ms):")
        print(results[['product_id', 'title', 'category', 'relevance_score']].to_string(index=False))
    else:
        print("Index not found. Please run index.py first.")
