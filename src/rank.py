import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

def train_ranker(data_path="data/ranking_data.parquet", model_path="data/ranker.lgb"):
    if not os.path.exists(data_path):
        print(f"Data {data_path} not found.")
        return

    print("Loading ranking data...")
    df = pd.read_parquet(data_path)
    
    # Sort by user_id for group-based ranking
    df = df.sort_values("user_id")
    
    features = ["cosine_sim", "price", "price_diff", "is_category_match"]
    X = df[features]
    y = df["label"]
    
    # Calculate group sizes
    groups = df.groupby("user_id").size().tolist()
    
    print(f"Training LightGBM Ranker on {len(df)} samples...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        importance_type='gain'
    )
    
    ranker.fit(
        X, y,
        group=groups,
        eval_at=[5, 10]
    )
    
    print(f"Saving ranker to {model_path}...")
    ranker.booster_.save_model(model_path)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': ranker.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(importance)
    print("Done!")

if __name__ == "__main__":
    train_ranker()
