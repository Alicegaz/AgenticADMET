# Solution for ASAP Antiviral ADMET 2025 challenge on Polaris
See [report](https://docs.google.com/document/d/16y9EtElTmJ-p0bAYjJaNgkU8WP-CYz77lhHB2oxkTc4/edit?usp=sharing) for understanding why the submission is so simple. Spoiler: because what is left here is just Gemini fine-tuning over Google Cloud API!

## Install dependencies
Install Miniconda with [official instructions](https://docs.anaconda.com/miniconda/install/). Then set up the environment:
```bash
conda create -n admet -y python=3.11
conda activate admet
conda install -c conda-forge -y polaris
pip install -r requirements.txt
```

## Reproduce solution
Go to `notebooks` folder and apply notebooks one-by-one according to prefix numbers starting with `notebooks/1.1_download_polaris_data.ipynb`. Have fun!

## (Optional) all experiments
If you want to see all experiments inlcuding Reinforcement Learning with reasoning, Gemini/GPT fine-tuning attempts, Chemprop and RoBERTa training, go to the branch `submission-and-experiments`
