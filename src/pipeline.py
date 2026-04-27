import torch
import numpy as np
import faiss
import pandas as pd
import lightgbm as lgb
from src.model import TwoTowerModel
from src.query import Recommender
import os

class RecommendationPipeline:
    def __init__(self, 
                 model_path="data/two_tower.pth", 
                 index_path="data/item_index.faiss", 
                 ranker_path="data/ranker.lgb",
                 products_path="data/products.parquet",
                 interactions_path="data/interactions.parquet",
                 embeddings_path="data/embeddings.npy"):
        
        print("Loading Pipeline components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Retrieval Components
        self.recommender = Recommender(model_path, index_path, products_path)
        
        # Ranking Component
        self.ranker = lgb.Booster(model_file=ranker_path)
        
        # Metadata and History for feature extraction
        self.df_interactions = pd.read_parquet(interactions_path)
        self.df_products = self.recommender.products
        self.embeddings = np.load(embeddings_path)
        self.id_to_idx = {pid: i for i, pid in enumerate(self.df_products['product_id'])}
        
        # Pre-calculate user profiles for fast lookup
        self.user_avg_prices = self.df_interactions.merge(self.df_products, on='product_id').groupby('user_id')['price'].mean().to_dict()
        self.user_fav_cats = self.df_interactions.merge(self.df_products, on='product_id').groupby('user_id')['category'].agg(lambda x: x.value_counts().index[0]).to_dict()
        self.user_histories = self.df_interactions.groupby('user_id')['product_id'].apply(list).to_dict()

    def get_user_embedding(self, user_id):
        if user_id not in self.user_histories:
            return None
        
        history_ids = self.user_histories[user_id]
        history_idxs = [self.id_to_idx[pid] for pid in history_ids if pid in self.id_to_idx]
        if not history_idxs:
            return None
            
        return np.mean(self.embeddings[history_idxs], axis=0)

    def recommend(self, user_id=None, image_feat=None, k=10):
        # 1. Determine Query Embedding
        user_feat = self.get_user_embedding(user_id) if user_id else None
        
        if user_feat is not None and image_feat is not None:
            # Combined Personalized Visual Search
            query_feat = (user_feat + image_feat) / 2
        elif user_feat is not None:
            query_feat = user_feat
        elif image_feat is not None:
            query_feat = image_feat
        else:
            # Fallback to random or popular (not implemented, using random for now)
            query_feat = np.random.randn(512).astype('float32')

        # 2. Stage 1: Retrieval (Top 50 candidates)
        candidates, retrieval_latency = self.recommender.recommend(query_feat, k=50)
        
        # 3. Stage 2: Re-Ranking
        if user_id in self.user_avg_prices:
            u_avg_p = self.user_avg_prices[user_id]
            u_fav_c = self.user_fav_cats[user_id]
        else:
            u_avg_p = self.df_products['price'].mean()
            u_fav_c = None
            
        # Extract features for ranking
        features = pd.DataFrame({
            "cosine_sim": candidates['relevance_score'],
            "price": candidates['price'],
            "price_diff": abs(candidates['price'] - u_avg_p),
            "is_category_match": (candidates['category'] == u_fav_c).astype(int) if u_fav_c else 0
        })
        
        candidates['ranking_score'] = self.ranker.predict(features)
        
        # Sort by ranking score
        final_results = candidates.sort_values("ranking_score", ascending=False).head(k)
        
        return final_results.to_dict(orient="records"), retrieval_latency
