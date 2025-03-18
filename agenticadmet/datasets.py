import numpy as np
import pandas as pd
from rdkit import Chem
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

from utils import CheckpointDownloader


def renumerate_smiles(smiles, random_seed=None):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m,ans)
    smiles = Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)

    return smiles


def randomly_mask_smiles(mol_sentence: str, mol_masking_val: float, mask_token: int) -> str:
    mask = torch.rand((len(mol_sentence),)) >= mol_masking_val
    masked_mol_sentence = ''.join([
        char if mask[i] else mask_token for i, char in enumerate(mol_sentence)
    ])

    return masked_mol_sentence


class RegressionDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            smiles_col: str,
            target_cols: list[str],
            tokenizer_name: str,
            max_length: int = 128,
            split: str | None = None,
            remove_all_nan_targets: bool = True,
            randomize_smiles: bool = False,
            mol_masking_prob: float = 0.3,
            mol_masking_val: float = 0.0,
            random_seed: int | None = None
    ):
        self.data_path = data_path
        self.smiles_col = smiles_col
        self.target_cols = target_cols
        self.max_length = max_length
        self.randomize_smiles = randomize_smiles
        self.mol_masking_prob = mol_masking_prob
        self.mol_masking_val = mol_masking_val
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.data = pd.read_csv(data_path)
        if split is not None:
            self.data = self.data[self.data['split'] == split]
        data_size = len(self.data)
        self.data = self.data[self.data[self.smiles_col].notna()]
        if remove_all_nan_targets:
            self.data = self.data[~self.data[self.target_cols].isna().all(axis=1)]
            print(f'{data_size - len(self.data)} out of {data_size} rows are removed due to missing values')
        self.data = self.data.reset_index(drop=True)

        self.smiles = self.data[self.smiles_col].tolist()
        self.targets = torch.from_numpy(self.data[self.target_cols].values).to(torch.float32)

        with CheckpointDownloader(tokenizer_name) as downloader:
            self.tokenizer = AutoTokenizer.from_pretrained(downloader.download_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        if self.randomize_smiles:
            smiles = renumerate_smiles(smiles, self.random_seed)
        if (self.mol_masking_val > 0.) and (np.random.rand() < self.mol_masking_prob):
            smiles = randomly_mask_smiles(smiles, self.mol_masking_val, self.tokenizer.mask_token)
        targets = self.targets[idx]
        weights = torch.ones(1, dtype=torch.float32)

        return {
            'smiles': smiles,
            'targets': targets,
            'weights': weights
        }
    
    def collate_fn(self, batch):
        smiles = [item['smiles'] for item in batch]
        targets = torch.stack([item['targets'] for item in batch])
        weights = torch.stack([item['weights'] for item in batch])

        tokenized = self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'targets': targets,
            'weights': weights
        }
