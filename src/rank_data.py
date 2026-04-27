import pandas as pd
import numpy as np
import torch
from src.query import Recommender
from tqdm import tqdm
import os

def generate_ranking_data(num_sample_users=1000, output_path="data/ranking_data.parquet"):
    if not os.path.exists("data/item_index.faiss"):
        print("Wait! FAISS index not found. This script requires Phase 3 execution.")
        return

    rec = Recommender()
    df_interactions = pd.read_parquet("data/interactions.parquet")
    df_products = pd.read_parquet("data/products.parquet")
    embeddings = np.load("data/embeddings.npy")
    id_to_idx = {pid: i for i, pid in enumerate(df_products['product_id'])}
    
    # Pre-compute user history and avg price
    user_histories = df_interactions.groupby('user_id')['product_id'].apply(set).to_dict()
    user_avg_prices = df_interactions.merge(df_products, on='product_id').groupby('user_id')['price'].mean().to_dict()
    user_fav_categories = df_interactions.merge(df_products, on='product_id').groupby('user_id')['category'].agg(lambda x: x.value_counts().index[0]).to_dict()

    sample_users = random.sample(list(user_histories.keys()), min(num_sample_users, len(user_histories)))
    
    ranking_rows = []
    
    print(f"Generating ranking features for {len(sample_users)} users...")
    for user_id in tqdm(sample_users):
        history_ids = list(user_histories[user_id])
        history_idxs = [id_to_idx[pid] for pid in history_ids if pid in id_to_idx]
        
        if not history_idxs: continue
        
        # User profile embedding
        user_feat_clip = np.mean(embeddings[history_idxs], axis=0)
        
        # Retrieve candidates
        candidates, _ = rec.recommend(user_feat_clip, k=50)
        
        user_avg_p = user_avg_prices[user_id]
        user_fav_cat = user_fav_categories[user_id]
        
        for _, row in candidates.iterrows():
            is_click = 1 if row['product_id'] in user_histories[user_id] else 0
            
            ranking_rows.append({
                "user_id": user_id,
                "product_id": row['product_id'],
                "cosine_sim": row['relevance_score'],
                "price": row['price'],
                "price_diff": abs(row['price'] - user_avg_p),
                "is_category_match": 1 if row['category'] == user_fav_cat else 0,
                "label": is_click
            })

    df_ranking = pd.DataFrame(ranking_rows)
    print(f"Saving {len(df_ranking)} ranking samples...")
    df_ranking.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    import random
    random.seed(42)
    generate_ranking_data()
