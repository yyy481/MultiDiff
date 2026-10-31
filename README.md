# MultiDiff：Multi-Scale and Uncertainty-Aware 3D Equivariant Diffusion for Pocket-Conditioned Drug Generation

MultiDiff is a pocket-conditioned 3D molecular generation framework built around an E(3)-equivariant diffusion model. The repository contains the diffusion generator, multi-scale protein–ligand graph construction, a hybrid uncertainty-aware property surrogate, an adaptive multi-objective reward, and reward-guided fine-tuning utilities.

## Requirements

### Recommended platform

- Linux
- NVIDIA GPU with a CUDA version compatible with the installed PyTorch build
- Conda or Mamba
- AutoDock Vina for docking-based evaluation

### Create the environment

The supplied `env.yml` is based on Python 3.7, PyTorch 1.13.1, and CUDA 11.6:

```bash
conda env create -f env.yml
conda activate MultiDiff
```
Alternatively, create a clean Python 3.7 environment and install the pinned Python dependencies from requirements.txt:

```bash
conda create -n MultiDiff python=3.7 -y
conda activate MultiDiff

conda install -c conda-forge rdkit=2018.09.1 openbabel=3.1.0 -y
python -m pip install -r requirements.txt
```
The PyTorch Geometric extensions in requirements.txt are tied to PyTorch 1.13 and CUDA 11.6. When using a different PyTorch or CUDA build, install matching versions of torch-scatter, torch-cluster, torch-sparse, and torch-spline-conv.
AutoDock Vina is required for docking-based evaluation. The Python dependencies used by the evaluation pipeline are included in requirements.txt.

Install the docking-related packages used by the evaluation pipeline:

```bash
python -m pip install meeko==0.1.dev3 pdb2pqr vina==1.2.2
python -m pip install "git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3"
```

The source also imports `py3Dmol`; install it when using visualization utilities:

```bash
python -m pip install py3Dmol
```

Do not rely on unpublished local wheel paths. For `torch-scatter`, `torch-cluster`, and the remaining PyTorch Geometric extensions, install builds that match the selected PyTorch and CUDA versions. A tested lock file or Docker image should be provided with the final release.

## Required data and model assets

Before running training, sampling, or evaluation, place the required data and checkpoints at the following paths:
```text
data/
├── PDBbind_v2020_pocket10/
│   ├── index.pkl
│   ├── <pocket files>
│   └── <ligand files>
├── PDBbind_pocket10_split.pt
└── PDBbind_v2020/                 # needed by some docking/evaluation modes

ckpt/
├── trained.pt                     # pretrained diffusion generator
└── bond_trained.pt                # bond predictor

surrogate/checkpoints/
├── ensemble_meta.pt
├── surrogate_model_0.pt
├── surrogate_model_1.pt
├── surrogate_model_2.pt
├── surrogate_model_3.pt
└── surrogate_model_4.pt
```

ProteinLigandDataset reads index.pkl from the raw pocket directory and creates a sibling LMDB directory automatically when the processed database is absent. The preprocessing pipeline should generate standardized ligand files, pocket files, index.pkl, and the fixed train/validation/test split before model training.

## Training

### Diffusion generator

```bash
python scripts/train.py \
  --config configs/train/train.yml \
  --device cuda:0 \
```

### Bond predictor

```bash
python scripts/train_bond.py \
  --config configs/train/train_bond.yml \
  --device cuda:0 \
```
## Reward-guided fine-tuning

The following command performs reward-guided fine-tuning using the pretrained diffusion generator and the five-member property-surrogate ensemble:

```bash
python rl_finetune/train_rl.py \
  --device cuda:0 \
  --checkpoint ckpt/trained.pt \
  --surrogate_dir surrogate/checkpoints \
  --pocket sample/3ztx_pocket.pdb \
  --save_dir rl_finetune/checkpoints \
  --max_iters 2000 \
  --mols_per_iter 32 \
  --lr 5e-6 \
  --reward_temp 0.5
```
The fine-tuned checkpoint is saved rl_finetune/checkpoints/.

For sampling with the fine-tuned generator, set the model checkpoint in the sampling configuration to:

```text
model:
  checkpoint: rl_finetune/checkpoints/rl.pt
```

## Sampling

Edit `configs/sample/sample.yml` to select the generator checkpoint, bond-predictor checkpoint, target pocket, generation mode, number of molecules, and property conditions.

```bash
python scripts/sample.py \
  --config configs/sample/sample.yml \
  --device cuda:0 \
  --outdir outputs
```

The provided pocket example can be used directly:

```yaml
model:
  checkpoint: ckpt/trained.pt
  target: sample/3ztx_pocket.pdb
  gen_mode: denovo

bond_predictor: ckpt/bond_trained.pt
```
To sample from the reward-guided model, replace ckpt/trained.pt with rl_finetune/checkpoints/rl.pt.

## Evaluation

Set `sample_path` and the dataset paths in `configs/eval/eval.yml`, then run:

```bash
python scripts/evaluate.py --config configs/eval/eval.yml
```

Supported docking modes in the evaluator include:

```text
qvina | vina_score | vina_dock | none
```
For reproducible comparisons, use the same test-pocket split, number of generated molecules per pocket, molecular reconstruction procedure, and docking configuration for all methods.

