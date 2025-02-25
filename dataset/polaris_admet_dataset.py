import numpy as np
import random
from datasets import Dataset, DatasetDict
import json

# DeepSeek system prompt for GRPO based training
# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )

SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

# Function to structure the training data
def make_conversation(example):
    """Convert dataset examples into conversation format."""
    return {
        "ground_truth": example["solution"],
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def split_dict(data, train_ratio=0.7, test_ratio=0.15, valid_ratio=0.15, seed=None):
    # Ensure ratios sum to 1
    assert abs((train_ratio + test_ratio + valid_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    keys = list(data.keys())
    random.shuffle(keys)
    
    num_samples = len(keys)
    train_end = int(train_ratio * num_samples)
    test_end = train_end + int(test_ratio * num_samples)
    
    # Split keys
    train_keys = keys[:train_end]
    test_keys = keys[train_end:test_end]
    valid_keys = keys[test_end:]
    
    # Build subset dictionaries
    train_data = {k: data[k] for k in train_keys}
    test_data = {k: data[k] for k in test_keys}
    valid_data = {k: data[k] for k in valid_keys}
    
    return train_data, test_data, valid_data

def flatten_properties(data, params):
    # dct = {"MLM": "MLM in uL/min/mg (Mouse Liver Microsomal stability, a stability assay that tests how quickly a molecule gets broken down by mouse liver microsomes, a useful assay that can be used as an estimate on how long a molecule will reside in the mouse body before it gets cleared.)",
    #     "HLM": "HLM in uL/min/mg (Human Liver Microsomal stability, a stability assay that tests how quickly a molecule gets broken down by human liver microsomes, a useful assay that can be used as an estimate on how long a molecule will reside in the human body before it gets cleared.)", 
    #     "KSOL": "KSOL in uM (Solubility, solubility is essential for drug molecules: this heavily affects the pharmacokinetic and dynamics ('PKPD') of the molecule in the human body.)",
    #     "LogD": "LogD (Lipophilicity, like solubility - but then in fatty tissue - LogD is a measure of a molecule's lipophilicity, or how well it dissolves in fat, LogD is calculated by comparing a molecule's solubility in octanol, a fat-like substance, to its solubility in water.)",
    #     "MDR1-MDCKII": "MDR1-MDCKII in 10^-6 cm/s (Cell permeation, MDCKII-MDR1 is a cell line that's used to model cell permeation i.e. how well drug compounds will permeate cell layers. For coronaviruses this is a critical endpoint because there is increasing evidence that afflictions such as long-covid are caused by (remnant) virus particles in the brain, and blood-brain-barrier (BBB) permeation is critical for drug candidates to reach the brain.)"
    #     }
    params = set(params)
    dct = {"MLM": "MLM in uL/min/mg (Mouse Liver Microsomal stability)",
        "HLM": "HLM in uL/min/mg (Human Liver Microsomal stability)", 
        "KSOL": "KSOL in uM (Solubility)",
        "LogD": "LogD (Lipophilicity, like solubility - but then in fatty tissue - LogD is a measure of a molecule's lipophilicity)",
        "MDR1-MDCKII": "MDR1-MDCKII in 10^-6 cm/s (Cell permeation)"
        }
    polaris_dataset_hold = []
    for k, v in data.items():
        for v_name, v_i in v.items():
            if not np.isnan(v_i) and v_name in params:
                polaris_dataset_hold.append({"solution": v_i, "problem": f"The numerical value of {dct[v_name]} of the small molecule given it's SMILES '{k}' is", "property": v_name})
    return polaris_dataset_hold

def load_polaris_dataset(params=["MLM", "HLM", "KSOL", "LogD", "MDR1-MDCKII"]):
    """Load and prepare the mathematics dataset."""
    with open("polaris-antiviral-admet-2025.json", "r") as f:
        polaris_dataset = json.load(f)

    train, test, valid = split_dict(polaris_dataset, seed=42)
    
    train, test, valid = flatten_properties(train, params=params), flatten_properties(test, params=params), flatten_properties(valid, params=params)

    # Convert splits to Datasets
    train_dataset = Dataset.from_list(train)
    test_dataset = Dataset.from_list(test)
    valid_dataset = Dataset.from_list(valid)



    # Combine into DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "validation": valid_dataset,
    })
    
    # Apply conversation format
    for split in dataset:
        dataset[split] = dataset[split].map(make_conversation)

    # for split in dataset:
    #     if "messages" in dataset[split].column_names:
    #         dataset[split] = dataset[split].remove_columns("messages")
    
    # for l in dataset["train"]:
    #     print(l)
    return dataset