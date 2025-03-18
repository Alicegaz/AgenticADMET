import numpy as np
import random
from datasets import Dataset, DatasetDict
import json
from pathlib import Path
from functools import partial
from .rules import rules, rules_long, rules_v2, rules_v3, rules_v4, rules_v5
import os

DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

RULES_MAPPING = {
    "rules_v3": rules_v3,
    "rules": rules,
    "rules_v2": rules_v2,
    "rules_long": rules_long,
    "rules_v4": rules_v4,
    "rules_v5": rules_v5
    # Add other rule sets if needed
}

dct = {"MLM": "is Mouse Liver Microsomal stability measured in uL/min/mg.",
    "HLM": "is Human Liver Microsomal stability measured in uL/min/mg.", 
    "KSOL": "is Solubility measured in uM.",
    "LogD": "is Lipophilicity, like solubility but then in fatty tissue. LogD is a measure of a molecule's lipophilicity.",
    "MDR1-MDCKII": "is Cell permeation measured in 10^-6 cm/s."
    }

SYSTEM_PROMPT = lambda x, rules_prompt_name: f"""You are an experienced Chemist that provides well-reasoned and detailed responses and excells at extimating ADME properties of molecules, especially {x}. {x} {dct[x]}
User asks you to estimate and predict {x} for a small molecule, you first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>. Put the final {x} prediction in the following format: \\boxed{{RESULT}}! Where RESULT is just the number in float or expression that is a final {x} prediction.
Here are some rules that might help you to predict {x}:
{RULES_MAPPING[rules_prompt_name]}
"""

# Function to structure the training data
def make_conversation(example, property, rules_prompt_name, system_prompt_fn = SYSTEM_PROMPT):
    """Convert dataset examples into conversation format."""
    return {
        "ground_truth": example["solution"],
        "prompt": [
            {"role": "system", "content": system_prompt_fn(property, rules_prompt_name)},
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

problem_template = lambda v_name, k, properties_i: f"What is the numerical value of {v_name} of the '{k}'? You may need some properties for this molecule from RDKiT for your calculations: {properties_i}"

def flatten_properties(data, params, problem_template_fn=problem_template, properties=False):
    params = set(params)

    polaris_dataset_hold = []
    for k, v in data.items():
        for v_name, v_i in v.items():
            properties_ = v["properties"].copy()
            properties_.pop("LogP")
            properties_.pop("Wildman_Crippen_LogP")
            properties_ = {k: round(v, 3) for k, v in properties_.items()}
            if v_name != "properties":
                if not np.isnan(v_i) and v_name in params:
                    polaris_dataset_hold.append({"solution": v_i,
                                                # "problem": f"The numerical value of {dct[v_name]} of the small molecule given it's SMILES '{k}' and properties it's {str(v['properties'])} is",
                                                "problem": problem_template_fn(v_name, k) if not properties else problem_template_fn(v_name, k, str(properties_)),
                                                "property": v_name, 
                                                "smiles": k})
    return polaris_dataset_hold

def load_polaris_dataset(params=["MLM", "HLM", "KSOL", "LogD", "MDR1-MDCKII"], rules_prompt_name="rules_v4", system_prompt_fn=SYSTEM_PROMPT, problem_template_fn=problem_template, properties=False, rewrite=False, shuffle=False, seed=42):
    """Load and prepare the mathematics dataset."""
    with open("polaris-antiviral-admet-2025_rdkit_properties.json", "r") as f: # "polaris-antiviral-admet-2025.json"
        polaris_dataset = json.load(f)

    if rewrite or not (DIR_PATH / "train_split.json").is_file():
        train, test, valid = split_dict(polaris_dataset, seed=seed)
        with open(DIR_PATH / "train_split.json", "w") as f:
            json.dump(train, f, indent=2)
        
        with open(DIR_PATH / "validation_split.json", "w") as f:
            json.dump(valid, f, indent=2)
        
        with open(DIR_PATH / "test_split.json", "w") as f:
            json.dump(test, f, indent=2)
    else:
        with open(DIR_PATH / "train_split.json", "r") as f:
            train = json.load(f)
        
        with open(DIR_PATH / "validation_split.json", "r") as f:
            valid = json.load(f)
        
        with open(DIR_PATH / "test_split.json", "r") as f:
            test = json.load(f)
    
    
    train, test, valid = flatten_properties(train, params=params, problem_template_fn=problem_template_fn, properties=properties), flatten_properties(test, params=params, problem_template_fn=problem_template_fn, properties=properties), flatten_properties(valid, params=params, problem_template_fn=problem_template_fn, properties=properties)

    # Convert splits to Datasets
    train_dataset = Dataset.from_list(train)
    test_dataset = Dataset.from_list(test)
    valid_dataset = Dataset.from_list(valid)

    if shuffle:
        train_dataset.shuffle(seed=seed)

    # Combine into DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "validation": valid_dataset,
    })
    
    # Apply conversation format
    for split in dataset:
        dataset[split] = dataset[split].map(partial(make_conversation, property="LogD", system_prompt_fn=system_prompt_fn, rules_prompt_name=rules_prompt_name))
    return dataset