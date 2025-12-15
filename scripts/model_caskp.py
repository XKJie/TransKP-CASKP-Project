import torch
import torch.nn as nn
from transformers import EsmModel
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem

# --- Helper Functions for Graph Creation ---
def get_atom_features(atom):
    possible_atoms = ['C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Co', 'Fe', 'Cu', 'Zn', 'Mg', 'Mn', 'Cr', 'Ni']
    features = [0] * (len(possible_atoms) + 1)
    try:
        idx = possible_atoms.index(atom.GetSymbol())
        features[idx] = 1
    except ValueError:
        features[-1] = 1
    return features

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    return [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC
    ]

def smiles_to_pyg_graph(smiles_string):
    """
    Converts a SMILES string into a PyTorch Geometric Data object.
    Returns None if the SMILES string is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features_list, dtype=torch.float)

        if mol.GetNumBonds() > 0:
            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.append((i, j))
                edge_indices.append((j, i))
                bond_features = get_bond_features(bond)
                edge_attrs.append(bond_features)
                edge_attrs.append(bond_features)

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except Exception:
        return None

# =====================================================================================
# ==                      TransKP Model and Components (Locked)                      ==
# =====================================================================================

class SubstrateGNN(nn.Module):
    """
    Graph Attention Network (GATv2) for processing substrate SMILES strings.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.1):
        super(SubstrateGNN, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv3 = GATv2Conv(hidden_dim * heads, output_dim, heads=1, dropout=dropout, concat=False)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(self.elu(self.conv1(x, edge_index)))
        x = self.dropout(self.elu(self.conv2(x, edge_index)))
        x = self.conv3(x, edge_index)
        
        if hasattr(data, 'batch') and data.batch is not None:
            from torch_geometric.nn import global_mean_pool
            graph_embedding = global_mean_pool(x, data.batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
            
        return graph_embedding

class FusionBlock(nn.Module):
    """
    A single block for cross-modal fusion, combining self-attention and cross-attention.
    """
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(FusionBlock, self).__init__()
        self.self_attn_protein = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_prot_to_sub = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn_protein = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, protein_emb, substrate_emb, protein_mask=None):
        protein_emb = self.norm1(protein_emb + self._sa_block(protein_emb, protein_mask))
        protein_emb = self.norm2(protein_emb + self._ca_block(protein_emb, substrate_emb))
        protein_emb = self.norm3(protein_emb + self.ffn_protein(protein_emb))
        return protein_emb
    
    def _sa_block(self, x, key_padding_mask):
        x, _ = self.self_attn_protein(x, x, x, key_padding_mask=key_padding_mask)
        return x

    def _ca_block(self, query, key_value):
        x, _ = self.cross_attn_prot_to_sub(query, key_value, key_value)
        return x

class DeepFusionKcatPredictor(nn.Module):
    """
    The TransKP model that integrates ESM-2 for protein sequences and a GNN for substrates.
    """
    def __init__(self, esm_model_name, gnn_input_dim, gnn_hidden_dim, gnn_heads, d_model, 
                 num_fusion_blocks, num_attn_heads, dim_feedforward, dropout=0.1):
        super(DeepFusionKcatPredictor, self).__init__()
        
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        self.protein_projection = nn.Linear(self.esm_model.config.hidden_size, d_model)
        self.gnn = SubstrateGNN(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, output_dim=d_model, heads=gnn_heads)
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(d_model, num_attn_heads, dim_feedforward, dropout) for _ in range(num_fusion_blocks)
        ])
        
        self.output_regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, input_ids, attention_mask, smiles_list):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        final_predictions = torch.zeros(batch_size, device=device, dtype=torch.float32)

        graphs = [smiles_to_pyg_graph(s) for s in smiles_list]
        valid_indices = [i for i, g in enumerate(graphs) if g is not None]

        if valid_indices:
            valid_graphs = [graphs[i] for i in valid_indices]
            graph_batch = Batch.from_data_list(valid_graphs).to(device)
            
            substrate_embedding = self.gnn(graph_batch).unsqueeze(1)

            valid_input_ids = input_ids[valid_indices]
            valid_attention_mask = attention_mask[valid_indices]
            esm_outputs = self.esm_model(input_ids=valid_input_ids, attention_mask=valid_attention_mask)
            protein_embedding = self.protein_projection(esm_outputs.last_hidden_state)

            fused_output = protein_embedding
            key_padding_mask = (valid_attention_mask == 0) 
            for block in self.fusion_blocks:
                fused_output = block(fused_output, substrate_embedding, protein_mask=key_padding_mask)

            masked_fused_output = fused_output * valid_attention_mask.unsqueeze(-1)
            summed_output = masked_fused_output.sum(dim=1)
            non_pad_count = valid_attention_mask.sum(dim=1, keepdim=True)
            pooled_output = summed_output / non_pad_count.clamp(min=1e-9)

            predicted_kcat = self.output_regressor(pooled_output).squeeze(-1)
            final_predictions[valid_indices] = predicted_kcat.to(torch.float32)

        return final_predictions

# =====================================================================================
# ==                      CASKP Model and Components (New Code)                      ==
# =====================================================================================

class AMSFF(nn.Module):
    """
    Adaptive Multi-Scale Feature Fusion (AMSFF) block.
    Extracts multi-scale features from sequence embeddings using 1D convolutions.
    """
    def __init__(self, d_model, d_multiscale, dropout=0.1):
        super(AMSFF, self).__init__()
        self.d_model = d_model
        
        self.conv_k3 = nn.Conv1d(d_model, d_multiscale, kernel_size=3, padding=1)
        self.conv_k9 = nn.Conv1d(d_model, d_multiscale, kernel_size=9, padding=4)
        self.conv_k21 = nn.Conv1d(d_model, d_multiscale, kernel_size=21, padding=10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.projection = nn.Linear(d_multiscale * 3, d_model)

    def forward(self, seq_embedding):
        x = seq_embedding.transpose(1, 2)
        
        h_local = self.relu(self.conv_k3(x))
        h_medium = self.relu(self.conv_k9(x))
        h_global = self.relu(self.conv_k21(x))
        
        h_multi_scale = torch.cat([h_local, h_medium, h_global], dim=1)
        h_multi_scale = h_multi_scale.transpose(1, 2)
        
        projected_features = self.dropout(self.projection(h_multi_scale))
        
        return projected_features

class HyperAttention(nn.Module):
    """
    HyperAttention Fusion block.
    Fuses sequence and structure embeddings using spatial cross-attention.
    """
    def __init__(self, d_model, struct_dim, num_heads=8, dropout=0.1):
        super(HyperAttention, self).__init__()
        self.d_model = d_model
        
        self.spatial_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.struct_projection = nn.Linear(struct_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, seq_embedding, struct_features):
        struct_kv = self.struct_projection(struct_features).unsqueeze(1)
        spatial_out, _ = self.spatial_attention(seq_embedding, struct_kv, struct_kv)
        fused_embedding = self.norm1(seq_embedding + spatial_out)
        
        return fused_embedding

class FullKcatPredictor(nn.Module):
    """
    The CASKP model, integrating ESM-2, AMSFF, HyperAttention, and a regressor.
    """
    def __init__(self, esm_model_name, struct_dim, d_model=256, d_multiscale=128, num_heads=8, dropout=0.1, use_amsff=True):
        super(FullKcatPredictor, self).__init__()
        self.use_amsff = use_amsff

        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        self.protein_projection = nn.Linear(self.esm_model.config.hidden_size, d_model)
        
        if self.use_amsff:
            self.amsff = AMSFF(d_model, d_multiscale, dropout)
        
        self.hyper_attention = HyperAttention(d_model, struct_dim, num_heads, dropout)
        
        self.output_regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, input_ids, attention_mask, struct_features):
        esm_outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        protein_embedding = self.protein_projection(esm_outputs.last_hidden_state)

        if self.use_amsff:
            seq_feat_multiscale = self.amsff(protein_embedding)
            fused_output = self.hyper_attention(seq_feat_multiscale, struct_features)
        else:
            fused_output = self.hyper_attention(protein_embedding, struct_features)
            
        masked_fused_output = fused_output * attention_mask.unsqueeze(-1)
        summed_output = masked_fused_output.sum(dim=1)
        non_pad_count = attention_mask.sum(dim=1, keepdim=True)
        pooled_output = summed_output / non_pad_count.clamp(min=1e-9)
        
        predicted_kcat = self.output_regressor(pooled_output)
        
        return predicted_kcat
