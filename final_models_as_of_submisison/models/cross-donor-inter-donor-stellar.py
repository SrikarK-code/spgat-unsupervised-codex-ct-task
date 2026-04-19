import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
import argparse
import scanpy as sc
from anndata import AnnData

# Import STELLAR models and utils
import models
from utils import entropy, MarginLoss
from datasets import GraphDataset

def build_spatial_edges(coords, n_neighbors=6):
    A = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='connectivity', include_self=False, n_jobs=-1)
    sources, targets = A.nonzero()
    return np.vstack((sources, targets))

def extract_stellar_data(df_subset, markers, le):
    X = df_subset[markers].values.astype(np.float32)
    y = le.transform(df_subset['Cell Type'])
    edges = build_spatial_edges(df_subset[['x', 'y']].values)
    return X, y, edges

# # --- PATCHED STELLAR CLASS ---
# class SafeSTELLAR:
#     def __init__(self, args, dataset):
#         self.args = args
#         self.dataset = dataset
        
#         # Ensure edges are strictly shaped [2, E] for PyG compatibility
#         if self.dataset.labeled_data.edge_index.shape[0] != 2:
#             self.dataset.labeled_data.edge_index = self.dataset.labeled_data.edge_index.t().contiguous()
#         if self.dataset.unlabeled_data.edge_index.shape[0] != 2:
#             self.dataset.unlabeled_data.edge_index = self.dataset.unlabeled_data.edge_index.t().contiguous()

#         args.input_dim = dataset.unlabeled_data.x.shape[-1]
#         self.model = models.Encoder(args.input_dim, args.num_heads).to(args.device)

#     def train_supervised(self, seed_model, optimizer):
#         seed_model.train()
#         ce = nn.CrossEntropyLoss()
#         labeled_x = self.dataset.labeled_data.to(self.args.device)
        
#         optimizer.zero_grad()
#         output, _, _ = seed_model(labeled_x)
#         loss = ce(output, labeled_x.y)
#         loss.backward()
#         optimizer.step()
#         return loss.item()

#     def est_seeds(self, seed_model, clusters, num_seed_class):
#         seed_model.eval()
#         with torch.no_grad():
#             unlabeled_graph = self.dataset.unlabeled_data.to(self.args.device)
#             output, _, _ = seed_model(unlabeled_graph)
#             prob = F.softmax(output, dim=1)
#             entr = -torch.sum(prob * torch.log(prob + 1e-8), 1).cpu().numpy()
        
#         entrs_per_cluster = []
#         for i in range(np.max(clusters)+1):
#             locs = np.where(clusters == i)[0]
#             entrs_per_cluster.append(np.mean(entr[locs]) if len(locs) > 0 else 0)
#         entrs_per_cluster = np.array(entrs_per_cluster)
        
#         novel_cluster_idxs = np.argsort(entrs_per_cluster)[-num_seed_class:] if num_seed_class > 0 else []
#         novel_label_seeds = np.zeros_like(clusters)
#         largest_seen_id = torch.max(self.dataset.labeled_data.y).item()
        
#         for i, idx in enumerate(novel_cluster_idxs):
#             novel_label_seeds[clusters == idx] = largest_seen_id + i + 1
#         return novel_label_seeds

#     def train_epoch(self, optimizer, m):
#         self.model.train()
#         bce = nn.BCELoss()
#         ce = MarginLoss(m=-m)

#         labeled_x = self.dataset.labeled_data.to(self.args.device)
#         unlabeled_x = self.dataset.unlabeled_data.to(self.args.device)
        
#         optimizer.zero_grad()
#         labeled_output, labeled_feat, _ = self.model(labeled_x)
#         unlabeled_output, unlabeled_feat, _ = self.model(unlabeled_x)
        
#         labeled_len = len(labeled_output)
#         batch_size = len(labeled_output) + len(unlabeled_output)
        
#         output = torch.cat([labeled_output, unlabeled_output], dim=0)
#         feat = torch.cat([labeled_feat, unlabeled_feat], dim=0)
#         prob = F.softmax(output, dim=1)
        
#         feat_norm = feat / torch.norm(feat, 2, 1, keepdim=True)
        
#         # Cosine distance contrastive pairs
#         target_np = labeled_x.y.cpu().numpy()
#         pos_pairs = []
#         for i in range(labeled_len):
#             idxs = np.where(target_np == target_np[i])[0]
#             if len(idxs) == 1:
#                 pos_pairs.append(idxs[0])
#             else:
#                 selec_idx = np.random.choice(idxs, 1)[0]
#                 while selec_idx == i:
#                     selec_idx = np.random.choice(idxs, 1)[0]
#                 pos_pairs.append(int(selec_idx))
        
#         # Compute cosine dist only for unlabel vs all (Memory Safe)
#         unlabel_cosine_dist = torch.mm(feat_norm[labeled_len:], feat_norm.t())
#         vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
#         pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
#         pos_pairs.extend(pos_idx)
        
#         pos_prob = prob[pos_pairs, :]
#         pos_sim = torch.bmm(prob.view(batch_size, 1, -1), pos_prob.view(batch_size, -1, 1)).squeeze()
        
#         bce_loss = bce(pos_sim, torch.ones_like(pos_sim))
        
#         unlabeled_ce_idx = torch.where(unlabeled_x.novel_label_seeds > 0)[0]
#         ce_idx = torch.cat((torch.arange(labeled_len).to(self.args.device), labeled_len + unlabeled_ce_idx))
#         target = torch.cat((labeled_x.y, unlabeled_x.novel_label_seeds.to(self.args.device)))
        
#         ce_loss = ce(output[ce_idx], target[ce_idx])
#         entropy_loss = entropy(torch.mean(prob, 0))
        
#         loss = 1 * bce_loss + 1 * ce_loss - 0.3 * entropy_loss
#         loss.backward()
#         optimizer.step()

#     def pred(self):
#         self.model.eval()
#         with torch.no_grad():
#             unlabeled_graph = self.dataset.unlabeled_data.to(self.args.device)
#             output, _, _ = self.model(unlabeled_graph)
#             prob = F.softmax(output, dim=1)
#             conf, pred = prob.max(1)
#         mean_uncert = 1 - torch.mean(conf).item()
#         return mean_uncert, pred.cpu().numpy()

#     def train(self):
#         unlabel_x = self.dataset.unlabeled_data.x.numpy()
#         adata = AnnData(unlabel_x)
#         sc.pp.neighbors(adata)
#         sc.tl.leiden(adata, resolution=1, flavor="igraph", n_iterations=2, directed=False)
#         clusters = adata.obs["leiden"].values.astype(int)

#         seed_model = models.FCNet(x_dim=self.args.input_dim, num_cls=torch.max(self.dataset.labeled_data.y).item() + 1).to(self.args.device)
#         seed_optimizer = optim.Adam(seed_model.parameters(), lr=1e-3, weight_decay=5e-2)
        
#         for epoch in range(20):
#             self.train_supervised(seed_model, seed_optimizer)
            
#         novel_label_seeds = self.est_seeds(seed_model, clusters, self.args.num_seed_class)
#         self.dataset.unlabeled_data.novel_label_seeds = torch.tensor(novel_label_seeds)
        
#         optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
#         for epoch in range(self.args.epochs):
#             mean_uncert, _ = self.pred()
#             self.train_epoch(optimizer, mean_uncert)



from torch_geometric.utils import subgraph
from torch_geometric.data import Data

class SafeSTELLAR:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        
        # Ensure edges are [2, E]
        if self.dataset.labeled_data.edge_index.shape[0] != 2:
            self.dataset.labeled_data.edge_index = self.dataset.labeled_data.edge_index.t().contiguous()
        if self.dataset.unlabeled_data.edge_index.shape[0] != 2:
            self.dataset.unlabeled_data.edge_index = self.dataset.unlabeled_data.edge_index.t().contiguous()

        args.input_dim = dataset.unlabeled_data.x.shape[-1]
        self.model = models.Encoder(args.input_dim, args.num_heads).to(args.device)

    # --- THE REGULARIZATION FIX ---
    def manual_cluster_data(self, graph_data, num_parts=100):
        """Replicates STELLAR's 100-subgraph regularization natively."""
        num_nodes = graph_data.x.shape[0]
        # perm = torch.randperm(num_nodes)
        perm = torch.randperm(num_nodes, device=graph_data.edge_index.device)
        chunk_size = (num_nodes // num_parts) + 1
        
        batches = []
        for i in range(0, num_nodes, chunk_size):
            subset = perm[i:i+chunk_size]
            sub_edge_index, _ = subgraph(subset, graph_data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
            
            # Rebuild the local subgraph
            sub_data = Data(x=graph_data.x[subset], edge_index=sub_edge_index)
            # if hasattr(graph_data, 'y'): sub_data.y = graph_data.y[subset]
            if getattr(graph_data, 'y', None) is not None: sub_data.y = graph_data.y[subset]
            if hasattr(graph_data, 'novel_label_seeds'): sub_data.novel_label_seeds = graph_data.novel_label_seeds[subset]
            
            # Store the original global indices so we can map predictions back
            sub_data.global_idx = subset 
            batches.append(sub_data)
        return batches

    def train_supervised(self, seed_model, optimizer):
        seed_model.train()
        ce = nn.CrossEntropyLoss()
        
        # Train on 100 isolated subgraphs
        labeled_batches = self.manual_cluster_data(self.dataset.labeled_data)
        
        for batch in labeled_batches:
            batch = batch.to(self.args.device)
            optimizer.zero_grad()
            output, _, _ = seed_model(batch)
            loss = ce(output, batch.y)
            loss.backward()
            optimizer.step()

    def est_seeds(self, seed_model, clusters, num_seed_class):
        seed_model.eval()
        unlabeled_batches = self.manual_cluster_data(self.dataset.unlabeled_data)
        
        # We must reconstruct the global entropy array
        global_entrs = torch.zeros(self.dataset.unlabeled_data.x.shape[0])
        
        with torch.no_grad():
            for batch in unlabeled_batches:
                batch = batch.to(self.args.device)
                output, _, _ = seed_model(batch)
                prob = F.softmax(output, dim=1)
                entr = -torch.sum(prob * torch.log(prob + 1e-8), 1)
                global_entrs[batch.global_idx] = entr.cpu()
                
        global_entrs = global_entrs.numpy()
        
        entrs_per_cluster = []
        for i in range(np.max(clusters)+1):
            locs = np.where(clusters == i)[0]
            entrs_per_cluster.append(np.mean(global_entrs[locs]) if len(locs) > 0 else 0)
        entrs_per_cluster = np.array(entrs_per_cluster)
        
        novel_cluster_idxs = np.argsort(entrs_per_cluster)[-num_seed_class:] if num_seed_class > 0 else []
        novel_label_seeds = np.zeros_like(clusters)
        largest_seen_id = torch.max(self.dataset.labeled_data.y).item()
        
        for i, idx in enumerate(novel_cluster_idxs):
            novel_label_seeds[clusters == idx] = largest_seen_id + i + 1
        return novel_label_seeds

    def train_epoch(self, optimizer, m):
        self.model.train()
        bce = nn.BCELoss()
        ce = MarginLoss(m=-m)

        labeled_batches = self.manual_cluster_data(self.dataset.labeled_data)
        unlabeled_batches = self.manual_cluster_data(self.dataset.unlabeled_data)
        
        # Zip them together (cycling unlabeled if labeled is longer)
        import itertools
        unlabel_iter = itertools.cycle(unlabeled_batches)
        
        for labeled_x in labeled_batches:
            unlabeled_x = next(unlabel_iter)
            
            labeled_x = labeled_x.to(self.args.device)
            unlabeled_x = unlabeled_x.to(self.args.device)
            
            optimizer.zero_grad()
            labeled_output, labeled_feat, _ = self.model(labeled_x)
            unlabeled_output, unlabeled_feat, _ = self.model(unlabeled_x)
            
            labeled_len = len(labeled_output)
            batch_size = len(labeled_output) + len(unlabeled_output)
            
            output = torch.cat([labeled_output, unlabeled_output], dim=0)
            feat = torch.cat([labeled_feat, unlabeled_feat], dim=0)
            prob = F.softmax(output, dim=1)
            
            feat_norm = feat / torch.norm(feat, 2, 1, keepdim=True)
            cosine_dist = torch.mm(feat_norm, feat_norm.t())

            target_np = labeled_x.y.cpu().numpy()
            pos_pairs = []
            for i in range(labeled_len):
                idxs = np.where(target_np == target_np[i])[0]
                if len(idxs) == 1:
                    pos_pairs.append(idxs[0])
                else:
                    selec_idx = np.random.choice(idxs, 1)[0]
                    while selec_idx == i:
                        selec_idx = np.random.choice(idxs, 1)[0]
                    pos_pairs.append(int(selec_idx))
            
            unlabel_cosine_dist = cosine_dist[labeled_len:, :]
            _, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
            pos_pairs.extend(pos_idx[:, 1].cpu().numpy().flatten().tolist())
            
            pos_prob = prob[pos_pairs, :]
            pos_sim = torch.bmm(prob.view(batch_size, 1, -1), pos_prob.view(batch_size, -1, 1)).squeeze()
            
            bce_loss = bce(pos_sim, torch.ones_like(pos_sim))
            
            unlabeled_ce_idx = torch.where(unlabeled_x.novel_label_seeds > 0)[0]
            ce_idx = torch.cat((torch.arange(labeled_len).to(self.args.device), labeled_len + unlabeled_ce_idx))
            target = torch.cat((labeled_x.y, unlabeled_x.novel_label_seeds))
            
            ce_loss = ce(output[ce_idx], target[ce_idx])
            entropy_loss = entropy(torch.mean(prob, 0))
            
            loss = 1 * bce_loss + 1 * ce_loss - 0.3 * entropy_loss
            loss.backward()
            optimizer.step()

    def pred(self):
        self.model.eval()
        
        # Prediction is done on the FULL graph to ensure seamless spatial mapping
        with torch.no_grad():
            unlabeled_graph = self.dataset.unlabeled_data.to(self.args.device)
            output, _, _ = self.model(unlabeled_graph)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            
        mean_uncert = 1 - torch.mean(conf).item()
        return mean_uncert, pred.cpu().numpy()

    def train(self):
        unlabel_x = self.dataset.unlabeled_data.x.numpy()
        adata = AnnData(unlabel_x)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=1, flavor="igraph", n_iterations=2, directed=False)
        clusters = adata.obs["leiden"].values.astype(int)

        seed_model = models.FCNet(x_dim=self.args.input_dim, num_cls=torch.max(self.dataset.labeled_data.y).item() + 1).to(self.args.device)
        seed_optimizer = optim.Adam(seed_model.parameters(), lr=1e-3, weight_decay=5e-2)
        
        for epoch in range(20):
            self.train_supervised(seed_model, seed_optimizer)
            
        novel_label_seeds = self.est_seeds(seed_model, clusters, self.args.num_seed_class)
        self.dataset.unlabeled_data.novel_label_seeds = torch.tensor(novel_label_seeds)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        for epoch in range(self.args.epochs):
            mean_uncert, _ = self.pred()
            self.train_epoch(optimizer, mean_uncert)



# --- EXECUTION SCRIPT ---
if __name__ == "__main__":
    print("Loading Base Data for STELLAR...")
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-seed-class', type=int, default=0)
    args = parser.parse_known_args()[0]
    args.cuda = torch.cuda.is_available()

    le = LabelEncoder()
    le.fit(df['Cell Type'].unique())
    
    # CRITICAL FIX: Dynamically set num_heads to match the exact number of cell classes
    args.num_heads = len(le.classes_)
    print(f"Set STELLAR output classes to: {args.num_heads}")

    csv_file = "stellar_meso_vs_macro_baselines.csv"
    with open(csv_file, "w") as f:
        f.write("Prior_Type,Prior_Source,Test_Region,Total_ARI,Total_F1\n")

    anchor_region = "B004_Ascending"
    inter_donors = ["B008", "B012"]
    
    print(f"Extracting Intra-Donor Reference: {anchor_region}...")
    X_intra, y_intra, edges_intra = extract_stellar_data(df[df['unique_region'] == anchor_region], markers, le)

    inter_data = {}
    for d in inter_donors:
        print(f"Extracting Inter-Donor Reference: {d}...")
        df_d = df[df['donor'] == d]
        df_d = df_d.sample(n=min(len(df_d), 20000), random_state=43) 
        inter_data[d] = extract_stellar_data(df_d, markers, le)

    df_b004 = df[df['donor'] == 'B004']
    unique_regions = df_b004['unique_region'].unique()

    for region in unique_regions:
        if region == anchor_region: continue
        
        df_sub = df_b004[df_b004['unique_region'] == region]
        if len(df_sub) < 10: continue
        
        print(f"\n{'='*60}\nTesting STELLAR on {region}\n{'='*60}")
        
        unlabeled_X = df_sub[markers].values.astype(np.float32)
        true_labels_str = df_sub['Cell Type'].values
        unlabeled_edges = build_spatial_edges(df_sub[['x', 'y']].values)

        # A. Test Intra-Donor (Meso)
        print(f"  -> Testing Intra-Donor Prior ({anchor_region})...")
        dataset_intra = GraphDataset(X_intra, y_intra, unlabeled_X, edges_intra, unlabeled_edges)
        stellar_intra = SafeSTELLAR(args, dataset_intra)
        stellar_intra.train()
        _, results_intra = stellar_intra.pred()
        pred_str_intra = le.inverse_transform(results_intra)
        
        ari_intra = adjusted_rand_score(true_labels_str, pred_str_intra)
        f1_intra = f1_score(true_labels_str, pred_str_intra, average='weighted')
        with open(csv_file, "a") as f:
            f.write(f"INTRA_DONOR,{anchor_region},{region},{ari_intra:.4f},{f1_intra:.4f}\n")
        print(f"     [SUCCESS] ARI: {ari_intra:.4f} | F1: {f1_intra:.4f}")

        # B. Test Inter-Donor (Macro)
        for d in inter_donors:
            print(f"  -> Testing Inter-Donor Prior ({d})...")
            X_macro, y_macro, edges_macro = inter_data[d]
            dataset_macro = GraphDataset(X_macro, y_macro, unlabeled_X, edges_macro, unlabeled_edges)
            stellar_macro = SafeSTELLAR(args, dataset_macro)
            stellar_macro.train()
            _, results_macro = stellar_macro.pred()
            pred_str_macro = le.inverse_transform(results_macro)
            
            ari_macro = adjusted_rand_score(true_labels_str, pred_str_macro)
            f1_macro = f1_score(true_labels_str, pred_str_macro, average='weighted')
            with open(csv_file, "a") as f:
                f.write(f"INTER_DONOR,{d},{region},{ari_macro:.4f},{f1_macro:.4f}\n")
            print(f"     [SUCCESS] ARI: {ari_macro:.4f} | F1: {f1_macro:.4f}")

    print("\nFinished all STELLAR Meso vs Macro baselines.")