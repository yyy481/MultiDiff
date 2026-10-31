import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class PropertyScorer:
    """
    Normalizes raw property values to [0, 1] scores where 1 = optimal.
    Handles different optimization directions per property.
    """

    def __init__(self):
        # Normalization ranges (from PDBbind statistics)
        # For each property: (min_val, max_val, direction)
        #   direction: 'max' = higher is better, 'min' = lower is better, 'range' = optimal in a range
        self.config = {
            'qed': {
                'min': 0.0, 'max': 1.0,
                'direction': 'max',  # higher QED = better drug-likeness
            },
            'sa': {
                'min': 1.0, 'max': 10.0,
                'direction': 'min',  # lower SA score = easier to synthesize
                # Note: in MultiDiff, SA is normalized to [0,1] where higher = easier
                # We handle both conventions
            },
            'logp': {
                'direction': 'range',
                'optimal_center': 2.0,  # optimal logP for drugs
                'optimal_width': 1.0,  # 卤1 from center is good
                'sigma': 1.5,  # gaussian falloff width
            },
            'aff': {
                'min': 0.0, 'max': 15.0,
                'direction': 'max',  # higher affinity = better binding
            },
        }

    def score(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """
        Convert raw property value to [0, 1] score where 1 = optimal.

        Args:
            name: property name ('qed', 'sa', 'logp', 'aff')
            value: [B] tensor of raw values
        Returns:
            score: [B] tensor in [0, 1]
        """
        cfg = self.config[name]

        if cfg['direction'] == 'max':
            # Higher is better: linear normalization
            score = (value - cfg['min']) / (cfg['max'] - cfg['min'] + 1e-8)
            score = score.clamp(0, 1)

        elif cfg['direction'] == 'min':
            # Lower is better: inverted normalization
            score = 1.0 - (value - cfg['min']) / (cfg['max'] - cfg['min'] + 1e-8)
            score = score.clamp(0, 1)

        elif cfg['direction'] == 'range':
            # Optimal in a range: gaussian scoring
            center = cfg['optimal_center']
            sigma = cfg['sigma']
            score = torch.exp(-((value - center) ** 2) / (2 * sigma ** 2))

        return score


class AdaptiveWeights:
    """
    Dynamically adjusts weights based on recent batch performance.

    Formula:
        w獗?= (1 - s虅獗? / 危(1 - s虅鈧?

    With EMA smoothing to prevent oscillation:
        w_smoothed = 畏 路 w_prev + (1-畏) 路 w_new
    """

    def __init__(self, n_objectives=4, ema_decay=0.9):
        self.n_objectives = n_objectives
        self.ema_decay = ema_decay
        # Initialize with uniform weights
        self.weights = torch.ones(n_objectives) / n_objectives
        self.initialized = False

    def update(self, batch_scores: torch.Tensor):
        """
        Update weights based on current batch's normalized scores.

        Args:
            batch_scores: [B, n_objectives] scores in [0, 1]
        """
        # Mean score per objective across the batch
        mean_scores = batch_scores.mean(dim=0).detach().cpu()  # [n_objectives]

        # New weights: worse-performing objectives get higher weight
        gaps = 1.0 - mean_scores  # [n_objectives]
        gaps = gaps.clamp(min=0.01)  # prevent zero weights
        new_weights = gaps / gaps.sum()

        # EMA smoothing
        if self.initialized:
            self.weights = self.ema_decay * self.weights + (1 - self.ema_decay) * new_weights
        else:
            self.weights = new_weights
            self.initialized = True

        # Re-normalize after smoothing
        self.weights = self.weights / self.weights.sum()

    def get_weights(self, device='cpu') -> torch.Tensor:
        """Return current weights as tensor [n_objectives]."""
        return self.weights.to(device)


class DiversityCalculator:
    """
    Computes structural diversity reward using Tanimoto similarity
    of ECFP6 fingerprints.

    R_diversity(i) = 1 - mean(Tanimoto(fp_i, fp_k)) for k != i in batch
    """

    def __init__(self, radius=3, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

    def compute(self, smiles_list: list) -> torch.Tensor:
        """
        Compute diversity reward for each molecule in the batch.

        Args:
            smiles_list: list of SMILES strings [B]
        Returns:
            diversity: [B] tensor, higher = more diverse
        """
        if not HAS_RDKIT:
            return torch.ones(len(smiles_list))

        # Compute fingerprints
        fps = []
        valid_idx = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.n_bits
                )
                fps.append(fp)
                valid_idx.append(i)
            else:
                fps.append(None)

        B = len(smiles_list)
        diversity = torch.zeros(B)

        for i in range(B):
            if fps[i] is None:
                diversity[i] = 0.0
                continue

            sims = []
            for j in range(B):
                if i == j or fps[j] is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sims.append(sim)

            if len(sims) > 0:
                diversity[i] = 1.0 - np.mean(sims)
            else:
                diversity[i] = 1.0  # only molecule in batch

        return diversity


class MultiObjectiveReward:
    """
    Complete multi-objective reward function integrating:
      1. Adaptive weighted scoring
      2. Uncertainty penalty (from Surrogate Ensemble + ENN)
      3. Structural diversity regularization

    R_final = 伪 路 (危 w獗?路 s獗?路 exp(-位 路 蟽獗?蟽獗糭ref)) + 尾 路 R_diversity

    For invalid molecules: R_final = 0
    """

    # Reference uncertainty values for normalization
    # (from your test results, Table 2 mean values)
    SIGMA_REF = {
        'qed': 0.1368,
        'sa': 0.4752,
        'logp': 0.8024,
        'aff': 1.6638,
    }

    PROPERTY_NAMES = ['qed', 'sa', 'logp', 'aff']

    def __init__(
            self,
            surrogate_ensemble,
            device='cpu',
            alpha=0.8,  # weight for property score
            beta=0.2,  # weight for diversity
            lambda_unc=2.0,  # uncertainty penalty strength
            ema_decay=0.9,  # weight smoothing factor
    ):
        """
        Args:
            surrogate_ensemble: SurrogateEnsemble instance (from Innovation 2)
            device: 'cuda:0' or 'cpu'
            alpha: balance between property optimization and diversity
            beta: diversity reward weight
            lambda_unc: uncertainty penalty coefficient
            ema_decay: EMA decay for adaptive weight smoothing
        """
        self.surrogate = surrogate_ensemble
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.lambda_unc = lambda_unc

        self.scorer = PropertyScorer()
        self.adaptive_weights = AdaptiveWeights(
            n_objectives=len(self.PROPERTY_NAMES),
            ema_decay=ema_decay,
        )
        self.diversity_calc = DiversityCalculator()

        # History for logging
        self.reward_history = deque(maxlen=100)

    def compute(
            self,
            atom_type: torch.Tensor,
            pos: torch.Tensor,
            edge_index: torch.Tensor,
            bond_type: torch.Tensor,
            batch: torch.Tensor,
            smiles_list: list = None,
            is_valid: torch.Tensor = None,
    ) -> dict:
        """
        Compute the full multi-objective reward for a batch of molecules.

        Args:
            atom_type, pos, edge_index, bond_type, batch:
                molecular graph (same format as Surrogate model input)
            smiles_list: list of SMILES [B] for diversity calculation
            is_valid: [B] bool tensor, True if molecule is chemically valid

        Returns:
            dict with keys:
                'R_final':       [B] final reward (0 for invalid molecules)
                'R_predict':     [B] weighted property score (before uncertainty)
                'R_uncertainty': [B] uncertainty-penalized score
                'R_diversity':   [B] diversity reward
                'weights':       [4] current adaptive weights
                'details':       dict with per-property scores, predictions, uncertainties
        """
        B = int(batch.max().item()) + 1  # number of molecules

        # ---- Step 1: Get predictions + uncertainties from Surrogate ----
        with torch.no_grad():
            surr_result = self.surrogate.predict(
                atom_type, pos, edge_index, bond_type, batch
            )

        # ---- Step 2: Score each property ----
        scores = {}  # normalized to [0, 1], higher = better
        predictions = {}  # raw predictions
        sigmas = {}  # total uncertainty

        for name in self.PROPERTY_NAMES:
            pred_mean = surr_result[name]['mean']  # [B]
            sigma_total = surr_result[name]['std_total']  # [B]

            # Normalize score to [0, 1]
            score = self.scorer.score(name, pred_mean)

            scores[name] = score
            predictions[name] = pred_mean
            sigmas[name] = sigma_total

        # Stack scores: [B, 4]
        score_tensor = torch.stack(
            [scores[name] for name in self.PROPERTY_NAMES], dim=-1
        )

        # ---- Step 3: Adaptive weights ----
        self.adaptive_weights.update(score_tensor)
        weights = self.adaptive_weights.get_weights(device=self.device)  # [4]

        # ---- Step 4: Uncertainty-penalized weighted score ----
        # R = 危 w獗?路 s獗?路 exp(-位 路 蟽獗?/ 蟽獗糭ref)
        penalty_factors = []
        for i, name in enumerate(self.PROPERTY_NAMES):
            sigma_normalized = sigmas[name] / self.SIGMA_REF[name]
            penalty = torch.exp(-self.lambda_unc * sigma_normalized)
            penalty_factors.append(penalty)

        penalty_tensor = torch.stack(penalty_factors, dim=-1)  # [B, 4]

        # Weighted score with uncertainty penalty
        R_predict = (weights * score_tensor).sum(dim=-1)  # [B]
        R_uncertainty = (weights * score_tensor * penalty_tensor).sum(dim=-1)  # [B]

        # ---- Step 5: Diversity reward ----
        if smiles_list is not None and len(smiles_list) == B:
            R_diversity = self.diversity_calc.compute(smiles_list).to(self.device)
        else:
            R_diversity = torch.ones(B, device=self.device)

        # ---- Step 6: Final reward ----
        R_final = self.alpha * R_uncertainty + self.beta * R_diversity

        # Apply validity mask
        if is_valid is not None:
            R_final = R_final * is_valid.float().to(self.device)

        # ---- Logging ----
        self.reward_history.append(R_final.mean().item())

        return {
            'R_final': R_final,
            'R_predict': R_predict,
            'R_uncertainty': R_uncertainty,
            'R_diversity': R_diversity,
            'weights': weights,
            'details': {
                'scores': scores,
                'predictions': predictions,
                'sigmas': sigmas,
                'penalty_factors': {
                    name: pf for name, pf in zip(self.PROPERTY_NAMES, penalty_factors)
                },
            },
        }

    def get_reward_summary(self) -> str:
        """Return a formatted summary of recent rewards."""
        if len(self.reward_history) == 0:
            return "No rewards computed yet."

        recent = list(self.reward_history)
        w = self.adaptive_weights.get_weights()
        lines = [
            f"Recent R_final: mean={np.mean(recent):.4f}, "
            f"std={np.std(recent):.4f}, "
            f"min={np.min(recent):.4f}, max={np.max(recent):.4f}",
            f"Adaptive weights: " + ", ".join(
                [f"{name}={w[i].item():.3f}" for i, name in enumerate(self.PROPERTY_NAMES)]
            ),
        ]
        return " | ".join(lines)