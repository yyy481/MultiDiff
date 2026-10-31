"""
Uncertainty-Aware Property Prediction Surrogate Model
======================================================
Hybrid approach: Deep Ensemble + Evidential Neural Network

Each model in the ensemble outputs:
  - μ (prediction mean) via standard readout
  - (γ, ν, α, β) via evidential readout → aleatoric + epistemic uncertainty

Final uncertainty combines:
  - σ_ensemble: disagreement between 5 models (epistemic, from ensemble)
  - σ_aleatoric: inherent noise in the property (from ENN per model)
  - σ_epistemic_enn: model's own epistemic estimate (from ENN per model)

Fusion: σ_total = sqrt(σ²_ensemble + σ²_aleatoric + σ²_epistemic_enn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum


class MPNNLayer(nn.Module):
    """Single message-passing layer."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, node_h, edge_index, edge_attr):
        src, dst = edge_index
        N = node_h.size(0)
        msg_input = torch.cat([node_h[src], node_h[dst], edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)
        agg = scatter_sum(messages, dst, dim=0, dim_size=N)
        update_input = torch.cat([node_h, agg], dim=-1)
        node_h = node_h + self.update_mlp(update_input)
        node_h = self.layer_norm(node_h)
        return node_h


class EvidentialHead(nn.Module):
    """
    Evidential regression head for a single property.
    Outputs (γ, ν, α, β) parameterizing a Normal-Inverse-Gamma distribution.

    From: Amini et al., "Deep Evidential Regression", NeurIPS 2020.

    γ:     prediction mean
    ν:     virtual observation count (higher = more confident)
    α:     shape parameter (must be > 1)
    β:     scale parameter (must be > 0)

    Aleatoric uncertainty = β / (α - 1)
    Epistemic uncertainty = β / (ν * (α - 1))
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 4 output heads
        self.gamma_head = nn.Linear(hidden_dim, 1)   # mean (unbounded)
        self.nu_head = nn.Linear(hidden_dim, 1)       # virtual obs count (>0)
        self.alpha_head = nn.Linear(hidden_dim, 1)    # shape (>1)
        self.beta_head = nn.Linear(hidden_dim, 1)     # scale (>0)

    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            gamma: [B, 1] prediction mean
            nu:    [B, 1] > 0
            alpha: [B, 1] > 1
            beta:  [B, 1] > 0
        """
        h = self.shared(x)
        gamma = self.gamma_head(h)
        # Softplus ensures positivity, +1 for alpha ensures > 1
        nu = F.softplus(self.nu_head(h)) + 1e-6
        alpha = F.softplus(self.alpha_head(h)) + 1.0 + 1e-6  # α > 1
        beta = F.softplus(self.beta_head(h)) + 1e-6
        return gamma, nu, alpha, beta

    @staticmethod
    def compute_uncertainties(nu, alpha, beta):
        """
        Compute aleatoric and epistemic uncertainty from NIG parameters.

        Returns:
            aleatoric:  [B, 1] inherent data noise
            epistemic:  [B, 1] model uncertainty
        """
        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        return aleatoric, epistemic


class EvidentialLoss(nn.Module):
    """
    Negative log-likelihood loss for Normal-Inverse-Gamma distribution
    with evidence regularization.

    From: Amini et al., "Deep Evidential Regression", NeurIPS 2020.
    """

    def __init__(self, coeff=0.01):
        super().__init__()
        self.coeff = coeff  # regularization coefficient

    def forward(self, gamma, nu, alpha, beta, target):
        """
        Args:
            gamma, nu, alpha, beta: NIG parameters [B, 1]
            target: true values [B, 1]
        Returns:
            loss: scalar
        """
        # NIG negative log-likelihood
        omega = 2.0 * beta * (1.0 + nu)
        nll = (
            0.5 * torch.log(torch.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(nu * (target - gamma) ** 2 + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Evidence regularization: penalize high evidence on wrong predictions
        error = torch.abs(target - gamma)
        evidence = 2.0 * nu + alpha
        reg = error * evidence

        loss = nll.mean() + self.coeff * reg.mean()
        return loss


class PropertyMPNN(nn.Module):
    """
    MPNN-based molecular property predictor with dual output heads:
      1. Standard readout → point prediction μ
      2. Evidential readout → (γ, ν, α, β) → aleatoric + epistemic uncertainty
    """

    def __init__(
        self,
        num_atom_types=16,
        num_bond_types=5,
        node_dim=128,
        edge_dim=64,
        hidden_dim=256,
        num_layers=4,
        num_properties=4,
        dropout=0.1,
    ):
        super().__init__()
        self.num_properties = num_properties

        # Embeddings
        self.atom_embedding = nn.Embedding(num_atom_types + 1, node_dim)
        self.bond_embedding = nn.Embedding(num_bond_types + 1, edge_dim)

        # Distance encoding
        self.dist_expansion = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.edge_combine = nn.Linear(edge_dim * 2, edge_dim)

        # MPNN layers (shared backbone)
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Head 1: Standard readout (point prediction)
        self.readout = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_properties),
        )

        # Head 2: Evidential readout (one EvidentialHead per property)
        self.evidential_heads = nn.ModuleList([
            EvidentialHead(input_dim=node_dim, hidden_dim=hidden_dim // 2)
            for _ in range(num_properties)
        ])

    def _encode(self, atom_type, pos, edge_index, bond_type, batch):
        """Shared MPNN backbone → graph-level representation."""
        node_h = self.atom_embedding(atom_type)

        bond_h = self.bond_embedding(bond_type)
        src, dst = edge_index
        dist = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True)
        dist_h = self.dist_expansion(dist)
        edge_attr = self.edge_combine(torch.cat([bond_h, dist_h], dim=-1))

        for layer in self.mpnn_layers:
            node_h = layer(node_h, edge_index, edge_attr)

        graph_h = scatter_mean(node_h, batch, dim=0)  # [B, node_dim]
        return graph_h

    def forward(self, atom_type, pos, edge_index, bond_type, batch):
        """
        Returns:
            predictions: [B, num_properties] (standard point predictions)
        """
        graph_h = self._encode(atom_type, pos, edge_index, bond_type, batch)
        predictions = self.readout(graph_h)
        return predictions

    def forward_evidential(self, atom_type, pos, edge_index, bond_type, batch):
        """
        Returns both standard predictions and evidential parameters.

        Returns:
            predictions: [B, num_properties]
            evidential: list of 4 tuples, each (gamma, nu, alpha, beta) with shape [B, 1]
        """
        graph_h = self._encode(atom_type, pos, edge_index, bond_type, batch)

        # Standard prediction
        predictions = self.readout(graph_h)

        # Evidential prediction (one head per property)
        evidential = []
        for head in self.evidential_heads:
            gamma, nu, alpha, beta = head(graph_h)
            evidential.append((gamma, nu, alpha, beta))

        return predictions, evidential

    def get_uncertainties(self, atom_type, pos, edge_index, bond_type, batch):
        """
        Convenience method: run forward_evidential and return structured uncertainties.

        Returns:
            dict with keys for each property:
                'pred':      [B] standard prediction
                'evi_mean':  [B] evidential prediction (gamma)
                'aleatoric': [B] aleatoric uncertainty
                'epistemic': [B] epistemic uncertainty (from ENN)
        """
        predictions, evidential = self.forward_evidential(
            atom_type, pos, edge_index, bond_type, batch
        )

        result = {}
        prop_names = ['qed', 'sa', 'logp', 'aff']
        for i, name in enumerate(prop_names):
            gamma, nu, alpha, beta = evidential[i]
            aleatoric, epistemic = EvidentialHead.compute_uncertainties(nu, alpha, beta)
            result[name] = {
                'pred': predictions[:, i],
                'evi_mean': gamma.squeeze(-1),
                'aleatoric': aleatoric.squeeze(-1),
                'epistemic_enn': epistemic.squeeze(-1),
            }
        return result


class SurrogateEnsemble:
    """
    Deep Ensemble of PropertyMPNN models with hybrid uncertainty.

    Uncertainty fusion:
        σ_total = sqrt(σ²_ensemble + σ²_aleatoric + σ²_epistemic_enn)

    where:
        σ_ensemble:     std of predictions across 5 models (epistemic from ensemble)
        σ_aleatoric:    mean of aleatoric uncertainty across 5 models (from ENN)
        σ_epistemic_enn: mean of ENN epistemic uncertainty across 5 models
    """

    PROPERTY_NAMES = ['qed', 'sa', 'logp', 'aff']

    def eval(self):
        """Set all models to eval mode"""
        for m in self.models:
            m.eval()
        return self

    def __init__(self, models=None, device='cpu'):
        self.device = device
        self.models = models or []
        self.n_models = len(self.models)

    @classmethod
    def build(cls, n_models=5, device='cpu', **model_kwargs):
        models = []
        for i in range(n_models):
            torch.manual_seed(42 + i * 1000)
            model = PropertyMPNN(**model_kwargs).to(device)
            models.append(model)
        return cls(models=models, device=device)

    def save(self, save_dir):
        import os
        os.makedirs(save_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            path = os.path.join(save_dir, f'surrogate_model_{i}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_idx': i,
            }, path)
        meta_path = os.path.join(save_dir, 'ensemble_meta.pt')
        torch.save({
            'n_models': self.n_models,
            'property_names': self.PROPERTY_NAMES,
        }, meta_path)

    @classmethod
    def load(cls, save_dir, device='cpu', **model_kwargs):
        import os
        meta = torch.load(os.path.join(save_dir, 'ensemble_meta.pt'), map_location=device)
        n_models = meta['n_models']
        models = []
        for i in range(n_models):
            path = os.path.join(save_dir, f'surrogate_model_{i}.pt')
            model = PropertyMPNN(**model_kwargs).to(device)
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            models.append(model)
        return cls(models=models, device=device)

    @torch.no_grad()
    def predict(self, atom_type, pos, edge_index, bond_type, batch):
        """
        Run all ensemble models with evidential heads.

        Returns:
            dict for each property:
                'mean':           [B] ensemble mean prediction
                'std_ensemble':   [B] ensemble disagreement (epistemic from ensemble)
                'std_aleatoric':  [B] average aleatoric uncertainty (from ENN)
                'std_epistemic':  [B] average ENN epistemic uncertainty
                'std_total':      [B] fused total uncertainty
                                  = sqrt(ensemble² + aleatoric² + epistemic²)
        """
        all_preds = []
        all_aleatoric = {name: [] for name in self.PROPERTY_NAMES}
        all_epistemic = {name: [] for name in self.PROPERTY_NAMES}

        for model in self.models:
            model.eval()
            # Standard predictions
            pred = model(atom_type, pos, edge_index, bond_type, batch)
            all_preds.append(pred)

            # Evidential uncertainties
            unc = model.get_uncertainties(atom_type, pos, edge_index, bond_type, batch)
            for name in self.PROPERTY_NAMES:
                all_aleatoric[name].append(unc[name]['aleatoric'])
                all_epistemic[name].append(unc[name]['epistemic_enn'])

        # Stack predictions: [n_models, B, num_properties]
        all_preds = torch.stack(all_preds, dim=0)
        ens_mean = all_preds.mean(dim=0)   # [B, 4]
        ens_std = all_preds.std(dim=0)     # [B, 4]

        # Build result
        result = {}
        for i, name in enumerate(self.PROPERTY_NAMES):
            # Ensemble uncertainty
            sigma_ens = ens_std[:, i]

            # Average aleatoric across models
            ale_stack = torch.stack(all_aleatoric[name], dim=0)  # [n_models, B]
            sigma_ale = ale_stack.mean(dim=0).sqrt()  # mean variance -> std

            # Average epistemic (ENN) across models
            epi_stack = torch.stack(all_epistemic[name], dim=0)  # [n_models, B]
            sigma_epi = epi_stack.mean(dim=0).sqrt()

            # Fused total uncertainty
            sigma_total = torch.sqrt(sigma_ens ** 2 + sigma_ale ** 2 + sigma_epi ** 2)

            result[name] = {
                'mean': ens_mean[:, i],
                'std_ensemble': sigma_ens,
                'std_aleatoric': sigma_ale,
                'std_epistemic': sigma_epi,
                'std_total': sigma_total,
            }

        result['raw'] = all_preds
        return result

    def predict_single_mol(self, atom_type, pos, edge_index, bond_type):
        """Convenience for single molecule. Returns plain floats."""
        batch = torch.zeros(len(atom_type), dtype=torch.long, device=self.device)
        atom_type = atom_type.to(self.device)
        pos = pos.to(self.device)
        edge_index = edge_index.to(self.device)
        bond_type = bond_type.to(self.device)

        result = self.predict(atom_type, pos, edge_index, bond_type, batch)

        output = {}
        for name in self.PROPERTY_NAMES:
            output[name] = {
                'mean': result[name]['mean'].item(),
                'std_ensemble': result[name]['std_ensemble'].item(),
                'std_aleatoric': result[name]['std_aleatoric'].item(),
                'std_epistemic': result[name]['std_epistemic'].item(),
                'std_total': result[name]['std_total'].item(),
            }
        return output