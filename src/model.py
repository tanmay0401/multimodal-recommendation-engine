import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    """A simple MLP tower for either User or Item."""
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class TwoTowerModel(nn.Module):
    def __init__(self, clip_dim=512, latent_dim=128):
        super().__init__()
        self.user_tower = Tower(input_dim=clip_dim, output_dim=latent_dim)
        self.item_tower = Tower(input_dim=clip_dim, output_dim=latent_dim)
        
    def forward(self, user_features, item_features):
        """
        user_features: [batch_size, clip_dim] (already aggregated history)
        item_features: [batch_size, clip_dim]
        """
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        
        # Normalize for cosine similarity
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        return user_emb, item_emb

def contrastive_loss(user_emb, item_emb, temperature=0.07):
    """
    In-batch negative contrastive loss.
    Each user_emb[i] should be close to item_emb[i] and far from item_emb[j] (j!=i).
    """
    # Dot product similarity matrix [batch_size, batch_size]
    logits = torch.matmul(user_emb, item_emb.transpose(0, 1)) / temperature
    
    # Ground truth: diagonal indices
    labels = torch.arange(user_emb.size(0)).to(user_emb.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss
