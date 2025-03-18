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
