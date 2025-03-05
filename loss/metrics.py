import hashlib
import numpy as np
from collections import defaultdict
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse


def compute_metrics(eval_pred):
    """
    Compute metrics function compatible with Hugging Face Trainer.
    
    Args:
        eval_pred (EvalPrediction): Object containing predictions and label_ids.
            - predictions: Model predictions (usually logits or text)
            - label_ids: Ground truth labels
    
    Returns:
        dict: Dictionary of metrics
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    print(eval_pred.label_ids)
    
    inputs = getattr(eval_pred, 'inputs', None)
    
    if inputs is None:
        raise ValueError("This metrics function requires inputs that contain SMILES strings")
    
    smiles = inputs  # Replace with actual extraction logic based on your data format
    
    contents = []
    if isinstance(predictions[0], str):
        contents = predictions
    else:
        contents = [str(pred) for pred in predictions]
    
    # Group by SMILES
    smiles2conts = defaultdict(list)
    for content, gold_val, smiles_i, prompt_dict in zip(contents, labels, smiles, kwargs["prompts"]):
        answer_val = None
        if gold_val is not None:
            # Use your parsing function to extract the answer
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            if len(answer_parsed) > 0 and not isinstance(answer_parsed[0], str):
                answer_val = float(answer_parsed[0])
        
        smiles_hash = hashlib.blake2b(smiles_i.encode('utf-8'), digest_size=4).hexdigest()
        smiles2conts[smiles_hash].append({
            "answer_val": answer_val,
            "gold_val": gold_val,
            "completion": content,
            "answer_val": answer_val,
            "system_input": prompt_dict[0]["content"],
            "user_prompt": prompt_dict[1]["content"],
        }) 
    
    # Calculate metrics
    median_maes = []
    
    for k, v in smiles2conts.items():
        answers_g = [v_i["answer_val"] for v_i in v]
        valid_answers = [float(v_i) for v_i in answers_g if v_i is not None]
        
        # Calculate median MAE for this SMILES group
        if len(valid_answers) > 0:
            answer_median = np.median(valid_answers)
            # Assuming all items in the group have the same gold value
            gold_val = float(v[0]["gold_val"]) if v[0]["gold_val"] is not None else None
            if gold_val is not None:
                # mae_median = abs(gold_val - answer_median)
                mae_median = (gold_val - answer_median)**2
                median_maes.append(mae_median)
    
    # Compute final metrics
    metrics = {
        "median_mse": np.mean(median_maes) if median_maes else float('inf')
    }

    # if table is not None:
    #     for v_i in v:
    #         table.add_data(k, mae_median, v_i["mae"], v_i["completion"], v_i["system_input"], v_i["user_prompt"], v_i["answer_parsed"], v_i["answer_val"], v_i["gold_val"]) 

    
    return metrics