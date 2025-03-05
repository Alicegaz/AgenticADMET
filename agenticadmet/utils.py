from dataclasses import dataclass
from pathlib import Path
import shutil
from uuid import uuid4

from google.cloud import storage
from google.api_core.exceptions import NotFound

from chembl_structure_pipeline.standardizer import standardize_mol, get_parent_mol
from chembl_structure_pipeline.exclude_flag import exclude_flag
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from typing import Optional


def ECFP_from_smiles(
    smiles,
    R = 2,
    L = 2**10,
    use_chirality = False,
    as_array = True):
    
    molecule = AllChem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    
    fpgen = AllChem.GetMorganGenerator(
        radius=R,
        fpSize=L,
        includeChirality=use_chirality
    )
    
    feature_list = fpgen.GetFingerprint(molecule)

    if as_array:
        return np.array(feature_list, dtype=bool)
    else:
        return feature_list


def tanimoto_similarity(x, y):
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    intersection = x @ y.T
    union = x.sum(axis=1)[:, None] + y.sum(axis=1) - intersection
    sim = intersection / union

    return sim


LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()
UNCHARGER = rdMolStandardize.Uncharger(canonicalOrder=True)


def standardize(smiles: str) -> Optional[str]:
    """Standardize a molecule and return its SMILES and a flag indicating whether the molecule is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except:
        return None
    
    exclude = exclude_flag(mol, includeRDKitSanitization=False)
    if exclude:
        return None

    # Standardize with ChEMBL data curation pipeline. During standardization, the molecule may be broken
    try:
        # Choose molecule with largest component
        mol = LARGEST_FRAGMENT_CHOOSER.choose(mol)
        # Standardize with ChEMBL data curation pipeline. During standardization, the molecule may be broken
        mol = standardize_mol(mol)
        smiles = Chem.MolToSmiles(mol)
    except:
        return None

    # Check if molecule can be parsed by RDKit (in rare cases, the molecule may be broken during standardization)
    if Chem.MolFromSmiles(smiles) is None:
        return None
    
    return smiles


def standardize_cxsmiles(smiles):
    return Chem.MolToCXSmiles(Chem.MolFromSmiles(smiles))


@dataclass
class CheckpointParams:
    path: str
    module_from: Optional[str] = None
    module_to: Optional[str] = None
    strict: bool = True


class CheckpointDownloader():
    def __init__(self, path_or_url: str):
        self.path_or_url = path_or_url
        self.download_path = None

    def __enter__(self):
        if self.path_or_url.startswith("gs://"):
            self._download_from_gcs()

        return self

    def _download_from_gcs(self):
        self.download_path = Path(f"checkpoint-{uuid4()}")

        bucket_name, key = self.path_or_url[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        try:
            print(f'Downloading checkpoint from {self.path_or_url}...')
            if CheckpointDownloader.is_gcs_directory(bucket_name, key):
                # Download directory (not recursive)
                self.download_path.mkdir(parents=True, exist_ok=False)
                blobs = bucket.list_blobs(prefix=key)
                for blob in blobs:
                    obj_key = blob.name[len(key):].lstrip('/')
                    blob.download_to_filename(str(self.download_path / obj_key))
            else:
                blob = bucket.blob(key)
                blob.download_to_filename(str(self.download_path))
        except NotFound:
            raise FileNotFoundError(f"Object {key} does not exist in bucket {bucket_name}.")
        finally:
            client.close()

    def __exit__(self, exc_type, exc_value, traceback):
        # Remove checkpoint file
        if self.download_path is None:
            return
        elif self.download_path.is_dir():
            shutil.rmtree(self.download_path)
        else:
            self.download_path.unlink()

    @staticmethod
    def is_gcs_directory(bucket_name: str, key: str) -> bool:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=key))
        
        # Directory with one object is classified as a file
        return len(blobs) > 1
    
    @property
    def path(self):
        return self.download_path or self.path_or_url
