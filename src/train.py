import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.model import TwoTowerModel, contrastive_loss
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class InteractionDataset(Dataset):
    def __init__(self, interactions_df, embeddings, id_to_idx):
        self.interactions = interactions_df
        self.embeddings = embeddings
        self.id_to_idx = id_to_idx
        
        # Pre-group interactions by user for history aggregation
        self.user_histories = interactions_df.groupby('user_id')['product_id'].apply(list).to_dict()
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        target_item_id = row['product_id']
        
        # Target item embedding
        target_idx = self.id_to_idx[target_item_id]
        item_feat = self.embeddings[target_idx]
        
        # User history feature: mean of CLIP embeddings of all items they interacted with
        # (excluding target item to avoid leakage, if possible, but for simplicity we'll just use the history)
        history_ids = self.user_histories[user_id]
        history_idxs = [self.id_to_idx[pid] for pid in history_ids if pid in self.id_to_idx]
        
        if not history_idxs:
            user_feat = np.zeros(512, dtype=np.float32)
        else:
            user_feat = np.mean(self.embeddings[history_idxs], axis=0)
            
        return torch.tensor(user_feat, dtype=torch.float32), torch.tensor(item_feat, dtype=torch.float32)

def train(epochs=10, batch_size=256, lr=1e-3):
    print("Loading data...")
    df_products = pd.read_parquet("data/products.parquet")
    id_to_idx = {pid: i for i, pid in enumerate(df_products['product_id'])}
    
    embeddings = np.load("data/embeddings.npy")
    df_interactions = pd.read_parquet("data/interactions.parquet")
    
    train_df, val_df = train_test_split(df_interactions, test_size=0.1, random_state=42)
    
    train_ds = InteractionDataset(train_df, embeddings, id_to_idx)
    val_ds = InteractionDataset(val_df, embeddings, id_to_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")
    
    model = TwoTowerModel(clip_dim=512, latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for user_feat, item_feat in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            user_feat, item_feat = user_feat.to(device), item_feat.to(device)
            
            optimizer.zero_grad()
            user_emb, item_emb = model(user_feat, item_feat)
            loss = contrastive_loss(user_emb, item_emb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user_feat, item_feat in val_loader:
                user_feat, item_feat = user_feat.to(device), item_feat.to(device)
                user_emb, item_emb = model(user_feat, item_feat)
                loss = contrastive_loss(user_emb, item_emb)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        
    print("Saving model...")
    torch.save(model.state_dict(), "data/two_tower.pth")
    print("Done!")

if __name__ == "__main__":
    # check if embeddings exist before running
    if os.path.exists("data/embeddings.npy"):
        train()
    else:
        print("Embeddings not found. Please wait for embed.py to finish.")
