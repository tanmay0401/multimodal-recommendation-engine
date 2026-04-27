import pandas as pd
import random
from tqdm import tqdm
import os

def generate_users(num_users=10000, products_path="data/products.parquet", output_path="data/interactions.parquet"):
    print(f"Loading products from {products_path}...")
    df_products = pd.read_parquet(products_path)
    product_ids = df_products['product_id'].tolist()
    
    # Group products by category for themed history generation
    category_map = df_products.groupby('category')['product_id'].apply(list).to_dict()
    categories = list(category_map.keys())

    print(f"Generating interactions for {num_users} users...")
    interactions = []
    
    for i in tqdm(range(num_users)):
        user_id = f"USER_{i:05d}"
        
        # Each user has a "favorite" category (80% preference)
        fav_category = random.choice(categories)
        num_interactions = random.randint(5, 20)
        
        for _ in range(num_interactions):
            if random.random() < 0.8:
                # Pick from favorite category
                p_id = random.choice(category_map[fav_category])
            else:
                # Pick from any category
                p_id = random.choice(product_ids)
                
            interactions.append({
                "user_id": user_id,
                "product_id": p_id,
                "timestamp": pd.Timestamp.now() - pd.Timedelta(days=random.randint(0, 30))
            })

    df_interactions = pd.DataFrame(interactions)
    print(f"Saving {len(df_interactions)} interactions to {output_path}...")
    df_interactions.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    generate_users()
