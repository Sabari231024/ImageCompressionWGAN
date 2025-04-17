
import torch.nn as nn
import torch
import torch.nn.functional as F
class GumbelQuantize(nn.Module):
    def __init__(self, num_hiddens, n_embed, embedding_dim, straight_through=False, kld_scale=5e-5, init_tau=1.0, min_tau=0.1, anneal_rate=0.00005):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.straight_through = straight_through
        self.temperature = init_tau
        self.min_tau = min_tau
        self.anneal_rate = anneal_rate
        self.kld_scale = kld_scale
        
        # Projection and embedding
        self.proj = nn.Conv2d(256, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)
        
        # Orthogonal initialization for embeddings
        nn.init.orthogonal_(self.embed.weight)
        

    def forward(self, z):
        # Project input features to logits
        
        logits = self.proj(z)
        
        # Update temperature for annealing
        self.temperature = max(self.min_tau, self.temperature * (1 - self.anneal_rate))
        
        # Gumbel-Softmax for quantization
        hard = self.straight_through if self.training else True
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        
        # Map to continuous embedding space
        z_q = torch.matmul(soft_one_hot.permute(0, 2, 3, 1), self.embed.weight).permute(0, 3, 1, 2)
        
        # Compute KL divergence
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        
        # Get discrete indices
        indices = soft_one_hot.argmax(dim=1)
        
        return z_q, diff
