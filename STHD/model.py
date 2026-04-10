import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class STHD_SpGAT(nn.Module):
    def __init__(self, num_cells, num_classes, num_genes):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(num_cells, num_classes))
        self.S = nn.Parameter(torch.ones(num_cells, 1)) # ELMM localized scalar
        self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)

    def forward(self, X, Mu, Var, edge_index):
        P = F.softmax(self.W, dim=1)
        
        # ELMM Log-Likelihood: Scales dict intensity for stressed/necrotic cells
        Mu_scaled = Mu.unsqueeze(0) * self.S.unsqueeze(2) 
        F_mat = -0.5 * torch.sum(((X.unsqueeze(1) - Mu_scaled) ** 2) / Var, dim=2)
        ll_prot = torch.sum(P * F_mat) / X.shape[0]

        # SpGAT dynamic spatial penalty across physical boundaries
        _, alpha = self.gat(X, edge_index, return_attention_weights=True)
        row, col = alpha[0]
        
        ce_space = -torch.sum(P[row] * alpha[1] * torch.log(P[col] + 1e-8)) / X.shape[0]
        return ll_prot, ce_space, P