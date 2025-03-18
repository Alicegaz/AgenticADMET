## Solution for ASAP Antiviral ADMET 2025 challenge on Polaris
This repository is meant for reading purposes. It remains as an artifact of the work on Antiviral ADMET 2025 competition. During the competition, we tried to apply a recently introduced GRPO from [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) (Reinforcement Learning branch of work), but ended up with simpler LLM-based methods like Gemini fine-tuning as well as Chemprop and SMILES-based RoBERTa training.

If a reader wants to reproduce the solution for the competition, they should read this repo in the following order:
1. Check submission notebook `notebooks/3_submit.ipynb` to get the full picture of the final submission
2. Go to `notebooks/1.1_download_polaris_data.ipynb` and `notebooks/2.2_data_splits.ipynb` for data downloading, cleaning and splitting that was used for the final submission as well as all experiments except RL branch of work
3. Add external Biogen ADME data with `notebooks/2.2.1_prepare_external.ipynb`
4. Come back to the notebook `notebooks/3_submit.ipynb` to perform training and submission

## Install dependencies (main)
Install Miniconda with [official instructions](https://docs.anaconda.com/miniconda/install/). Then set up the environment:
```bash
conda create -n admet -y python=3.11
conda activate admet
conda install -c conda-forge -y polaris
pip install -r requirements.txt
```

## Install dependencies (for Reinforcement Learning experiments)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv openr1 --python 3.11 && source openr1/bin/activate && GIT_LFS_SKIP_SMUDGE=1 uv pip install -r requirements_rl.txt --upgrade pip --link-mode=copy  --no-build-isolation
```

### install rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```
