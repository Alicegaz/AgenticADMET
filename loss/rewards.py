# Import necessary libraries
import re

# Import math-related utilities
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import numpy as np
import math
from collections import defaultdict
import hashlib
from functools import lru_cache
#source openr1/bin/activate
import json
from functools import partial
import wandb


def accuracy_reward(completions, ground_truth=None, log_normalize=False, mae_thr=None, **kwargs):
    """
    Reward function to check if the model's response is mathematically 
    equivalent to the ground truth solution.
    Uses latex2sympy2 for parsing and math_verify for validation.
    """
    # print("completions", completions)
    #TODO: reward right niw is -inf;0, 0 for perfect match between groundtruth and prediciton, howether when solution is None (now we exluded such cases) we set reward to 0.5, which violates our current range
    #TODO: does the current range of -inf;0 
    # print(ground_truth, kwargs.keys())

    # Possible values
    # Cannot convert expression to float [3.5*to*4.5, '3.5\\text{ to }4.5']

    # Extract responses
    contents = [completion[0]["content"] for completion in completions] #TODO: why do we take 0
    rewards = []

    solutions = kwargs.get("solution") # Get solutions from kwargs
    property = kwargs.get("property")
    # smiles = kwargs.get("smiles")

    if solutions is None:
        return [0.5] * len(completions) # Return neutral reward if no solution
    
    for content, gold_val in zip(contents, solutions):
        
        if gold_val is not None:  # Check if parsing was successful
            # Parse the model's answer with relaxed normalization
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
                extraction_mode="first_match", # TODO: last match
            )

            # Reward 1.0 if correct, 0.0 if incorrect
            # reward = float(verify(answer_parsed, gold_parsed))

            # Evaluate symbolic expressions as floats
            # gold_val = float(gold_parsed.evalf())
            # answer_val = float(answer_parsed.evalf())
            if len(answer_parsed) > 0 and not isinstance(answer_parsed[0], str):
                answer_val = float(answer_parsed[0])
                # if log_normalize and property != "LogD" and (answer_val > 0.5):
                #     answer_val = np.log10(np.clip(answer_val, a_min=0.5, a_max=None))
                #     gold_val = np.log10(np.clip(gold_val, a_min=0.5, a_max=None))
                # else:
                #     reward = 0.5    # TODO: fix it for a more smooth value
                # reward = np.mean(np.abs(gold_val - answer_val))
                mae = np.abs(gold_val - answer_val)
                # mse = (gold_val - answer_val)**2
                # reward = np.clip(1-(1/6)*mse, 0, 1)
                reward = np.clip(1-(1/6)*mae, 0, 1)

                if mae_thr is not None:
                    reward = reward if mae <= mae_thr else 0
                print("parsed correctly", answer_val, gold_val, f"reward: {round(reward, 3)}, mae: {round(mae, 3)}") #, mse: {round(mse, 3)}")
            else:
                reward = 0
                if len(answer_parsed) > 0:
                    print("parsed not correctly", answer_parsed, type(answer_parsed[0]))
        else:
            # If ground truth cannot be parsed, assign neutral reward (0.5)
            reward = 0.5
            print("Warning: Gold solution is None:", gold_val)

        rewards.append(reward)
    # print(completions)
    
    return rewards

STEPS = 0

def compute_mae_v2(completions, ground_truth=None, log_normalize=False, table=None, text_table_current=None, **kwargs):
    global STEPS
    STEPS+=1
    contents = [completion[0]["content"] for completion in completions] #TODO: why do we take 0
    solutions = kwargs.get("solution") # Get solutions from kwargs
    smiles = kwargs.get("smiles")


    if solutions is None:
        return [0.5] * len(contents) # Return neutral reward if no solution
    smiles2conts = defaultdict(list)
    for content, gold_val, smiles_i, prompt_dict in zip(contents, solutions, smiles, kwargs["prompts"]):
        answer_val = None
        mse = 4.0
        mae = 2.0
        reward = 0.0
        if gold_val is not None:
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
                mse = (answer_val - float(gold_val))**2
                mae = np.abs(answer_val - float(gold_val))
                reward = np.clip(1-(1/6)*mse, 0, 1)
        
        smiles_hash = hashlib.blake2b(smiles_i.encode('utf-8'), digest_size=4).hexdigest()
        smiles2conts[smiles_hash].append({
                       "completion": content,
                       "answer_val": answer_val,
                       "answer_parsed": str(answer_parsed), 
                       "mse": mse,
                       "mae": mae,
                       "reward": reward,
                       "gold_val": str(gold_val),
                       "system_input": prompt_dict[0]["content"],
                       "user_prompt": prompt_dict[1]["content"],
                       }) 
    median_mses = []
    median_maes = []
    for k, v in smiles2conts.items():
        answers_g = [v_i["answer_val"] for v_i in v]
        answers_g = [float(v_i) for v_i in answers_g if v_i is not None and not np.isnan(v_i)]
        if len(answers_g) > 0:
            answer_median = np.median(answers_g)
            mse_median = (float(v[0]["gold_val"]) - answer_median)**2
            mae_median = np.abs(float(v[0]["gold_val"]) - answer_median)
            median_mses.append(mse_median)
            median_maes.append(mae_median)
        else:
            mse_median = 4.0
            mae_median = 2.0
            median_mses.append(4.0)
            median_maes.append(2.0)
        if table is not None:
            for v_i in v:
                table.add_data(k, STEPS, v_i["reward"], mae_median, v_i["mae"], v_i["completion"], v_i["system_input"], v_i["user_prompt"], v_i["answer_parsed"], v_i["answer_val"], v_i["gold_val"]) 
                text_table_current.add_data(k, v_i["reward"], mae_median, v_i["mae"], v_i["completion"], v_i["system_input"], v_i["user_prompt"], v_i["answer_parsed"], v_i["answer_val"], v_i["gold_val"])
    wandb.log({f"running_tables/training_samples_current_sample": text_table_current})
    return median_maes

# Implement Format Reward Function
def format_reward(completions, **kwargs):
    """
    Reward function to check if the completion has the correct format:
    <think>...</think> <answer>...</answer>.
    """
    # Define the regex pattern for the desired format
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    # Extract the content from each completion
    completion_contents = [completion[0]["content"] for completion in completions]

    # Check if each completion matches the pattern
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE)
               for content in completion_contents]

    # Reward 1.0 for correct format, 0.0 otherwise
    return [1.0 if match else 0.0 for match in matches]


def reasoning_steps_reward(completions, **kwargs):
    r"""
    Reward function to encourage clear step-by-step reasoning.
    It looks for patterns like "Step 1:", numbered lists, bullet points,
    and transition words.
    """
    # Regex pattern to find indicators of reasoning steps
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # Extract completion contents
    completion_contents = [completion[0]["content"] for completion in completions]

    # Count the number of reasoning step indicators in each completion
    matches = [len(re.findall(pattern, content, re.MULTILINE))
               for content in completion_contents]

    # Reward is proportional to the number of reasoning steps, maxing out at 1.0
    # We're using a "magic number" 3 here - encourage at least 3 steps for full reward
    return [min(1.0, count / 3) for count in matches]

def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1):
    """
    Returns a repetition penalty reward function. Penalizes repetitions of n-grams
    in the generated text.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        """Helper function to generate n-grams from text."""
        words = text.lower().split() # Lowercase and split into words
        return zip(*[words[i:] for i in range(ngram_size)]) # Create n-grams

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        Repetition penalty reward function.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "": # No penalty for empty completions
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size: # No penalty for short completions
                rewards.append(0.0)
                continue

            ngrams = set() # Use a set to store unique n-grams
            total = 0
            for ng in zipngram(completion, ngram_size): # Generate n-grams
                ngrams.add(ng) # Add n-gram to the set (duplicates are ignored)
                total += 1 # Count total n-grams

            # Calculate scaling factor: more repetition -> higher scaling
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty # Apply penalty based on scaling
            rewards.append(reward)
        return rewards
    return repetition_penalty_reward

# Utility function to get reward functions based on script arguments
def get_reward_functions(script_args, mae_thr=None, table=None, text_table_current=None):
    """
    Returns a list of reward functions based on the script arguments.
    """
    accuracy_reward_fn = partial(accuracy_reward, mae_thr=mae_thr)
    accuracy_reward_fn.__name__ = "accuracy"
    
    compute_mae_v2_fn = partial(compute_mae_v2, table=table, text_table_current=text_table_current)
    compute_mae_v2_fn.__name__ = "compute_mae_v2"
    reward_funcs_list = []
    reward_funcs_registry = {
        "accuracy": accuracy_reward_fn,  # Assuming accuracy_reward is defined in previous steps
        "format": format_reward,      # Assuming format_reward is defined in previous steps
        "reasoning_steps": reasoning_steps_reward, # Assuming reasoning_steps_reward is defined
        # "cosine": get_cosine_scaled_reward( # Assuming get_cosine_scaled_reward is defined
        #     min_value_wrong=script_args.cosine_min_value_wrong,
        #     max_value_wrong=script_args.cosine_max_value_wrong,
        #     min_value_correct=script_args.cosine_min_value_correct,
        #     max_value_correct=script_args.cosine_max_value_correct,
        #     max_len=script_args.cosine_max_len,
        # ),
        "repetition_penalty": get_repetition_penalty_reward( # Assuming get_repetition_penalty_reward is defined
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "compute_mae_v2": compute_mae_v2_fn
    }

    for func_name in script_args.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list


# # TODO: does it run on the entire list
# def compute_metric(completions, ground_truth=None, log_normalize=False, **kwargs):
#     """
#     Reward function to check if the model's response is mathematically 
#     equivalent to the ground truth solution.
#     Uses latex2sympy2 for parsing and math_verify for validation.
#     """
    
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []

#     solutions = kwargs.get("solution") # Get solutions from kwargs
#     property = kwargs.get("property")

#     if solutions is None:
#         return [0.5] * len(completions) # Return neutral reward if no solution
    
#     for content, gold_val in zip(contents, solutions):
        
#         if gold_val is not None:  # Check if parsing was successful
#             answer_parsed = parse(
#                 content,
#                 extraction_config=[
#                     LatexExtractionConfig(
#                         normalization_config=NormalizationConfig(
#                             nits=False,
#                             malformed_operators=False,
#                             basic_latex=True,
#                             equations=True,
#                             boxed="all",
#                             units=True,
#                         ),
#                         boxed_match_priority=0,
#                         try_extract_without_anchor=False,
#                     )
#                 ],
#                 extraction_mode="first_match",
#             )

#             try:
#                 answer_val = float(answer_parsed[0])
#                 if log_normalize and property != "LogD":
#                     answer_val = np.log10(answer_val+1)
#                     gold_val = np.log10(gold_val+1)
#                 reward = np.mean(np.abs(gold_val - answer_val))
#                 reward = np.clip(1-(1/6)*reward, 0, 1)
#                 print("parsed correctly", answer_val, gold_val)
#             except Exception as e:
#                 reward = 0
#                 if len(answer_parsed) > 0:
#                     print(e, answer_parsed)
#         else:
#             # If ground truth cannot be parsed, assign neutral reward (0.5)
#             reward = 0.5
#             print("Warning: Gold solution is None:", gold_val)

#         rewards.append(reward)

#     return rewards


# Precompile regex patterns
DIGIT_PATTERN = re.compile(r'^[-+]?\d*\.?\d*$')
NUMBER_PATTERN = re.compile(r'[-+]?\d*\.?\d+')

@lru_cache(maxsize=1024)
def hash_smile(smile):
    """Fast hashing with fixed digest size"""
    return hashlib.blake2b(smile.encode('utf-8'), digest_size=4).hexdigest()

def mae_not_reward(completions, ground_truth=None, log_normalize=False, MAX_ERROR=6.0, **kwargs):
    
    #TODO: add log normalize support
    #TODO: change based on max value of a property

    contents = [completion[0]["content"] for completion in completions] #TODO: why do we take 0

    solutions = kwargs.get("solution") # Get solutions from kwargs
    property = kwargs.get("property")
    smiles = kwargs.get("smiles")

    # Fast path for empty solutions
    if solutions is None:
        return [0.5] * len(contents)

    smile_groups = defaultdict(list)
    
    # Single pass through data with minimal processing
    for i, (content, gold_val, smile) in enumerate(zip(contents, solutions, smiles)):
        if gold_val is None:
            continue
            
        # Extract first number match directly - much faster than full parsing
        match = NUMBER_PATTERN.search(content)
        answer_val = float(match.group(0)) if match else None
        
        # Calculate error directly
        mae = abs(gold_val - answer_val) if answer_val is not None else MAX_ERROR
        
        # Group by hashed SMILE
        smile_hash = hash_smile(smile)
        smile_groups[smile_hash].append((answer_val, gold_val, mae))
    
    # Calculate median MAEs
    median_maes = []
    for values in smile_groups.values():
        # Extract valid answers only once
        valid_answers = [v[0] for v in values if v[0] is not None]
        
        if valid_answers:
            # Use numpy's efficient median calculation
            answer_median = np.median(valid_answers)
            gold = values[0][1]  # All gold values in a group are the same
            mae_median = abs(gold - answer_median)
        else:
            mae_median = MAX_ERROR
            
        median_maes.append(mae_median)
    
    return median_maes

# def compute_mae_v2(completions, ground_truth=None, log_normalize=False, table=None, **kwargs):
#     contents = [completion[0]["content"] for completion in completions] #TODO: why do we take 0
#     solutions = kwargs.get("solution") # Get solutions from kwargs
#     smiles = kwargs.get("smiles")


#     if solutions is None:
#         return [0.5] * len(contents) # Return neutral reward if no solution
#     smiles2conts = defaultdict(list)
#     for content, gold_val, smiles_i in zip(contents, solutions, smiles):
#         answer_val = None
#         mse = 4
#         if gold_val is not None:
#             answer_parsed = parse(
#                 content,
#                 extraction_config=[
#                     LatexExtractionConfig(
#                         normalization_config=NormalizationConfig(
#                             nits=False,
#                             malformed_operators=False,
#                             basic_latex=True,
#                             equations=True,
#                             boxed="all",
#                             units=True,
#                         ),
#                         boxed_match_priority=0,
#                         try_extract_without_anchor=False,
#                     )
#                 ],
#                 extraction_mode="first_match",
#             )

#             if len(answer_parsed) > 0 and not isinstance(answer_parsed[0], str):
#                 answer_val = float(answer_parsed[0])
#                 mse = (answer_val - float(gold_val))**2
        
#         smiles_hash = hashlib.blake2b(smiles_i.encode('utf-8'), digest_size=4).hexdigest()
#         smiles2conts[smiles_hash].append({
#                        "answer_val": answer_val,
#                        "mse": mse,
#                        "gold_val": str(gold_val)
#                        }) 
#     median_mses = []
#     median_maes = []
#     for k, v in smiles2conts.items():
#         answers_g = [v_i["answer_val"] for v_i in v]
#         answers_g = [float(v_i) for v_i in answers_g if v_i is not None and not np.isnan(v_i)]
#         if len(answers_g) > 0:
#             answer_median = np.median(answers_g)
#             mse_median = (float(v[0]["gold_val"]) - answer_median)**2
#             mae_median = np.abs(float(v[0]["gold_val"]) - answer_median)
#             median_mses.append(mse_median)
#             median_maes.append(mae_median)
#         else:
#             mse_median = 4.0
#             mae_median = 2.0
#             median_mses.append(mse_median)
#             median_maes.append(mae_median)
#         if table is not None:
#             for v_i in v:
#                 table.add_data(k, mae_median, v_i["mae"], v_i["completion"], v_i["system_input"], v_i["user_prompt"], v_i["answer_parsed"], v_i["answer_val"], v_i["gold_val"]) 

#     return median_maes


# If you need to support batch-based metrics with batch_eval_metrics=True
def compute_metrics_with_batch_support(eval_pred, compute_result=False, log_normalize=False, MAX_ERROR=6.0):
    """
    Batch-supporting compute metrics function for Hugging Face Trainer.
    
    Args:
        eval_pred (EvalPrediction): Object containing predictions and label_ids
        compute_result (bool): If True, return the final metrics; if False, accumulate batch statistics
        log_normalize (bool, optional): Whether to log normalize values. Defaults to False.
        MAX_ERROR (float, optional): Maximum error value. Defaults to 6.0.
            
    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Initialize accumulators the first time
    if not hasattr(compute_metrics_with_batch_support, "smile_groups"):
        compute_metrics_with_batch_support.smile_groups = defaultdict(list)
        compute_metrics_with_batch_support.valid_predictions = 0
        compute_metrics_with_batch_support.total_predictions = 0
    
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Parse input data (similar to non-batch version)
    is_text_output = isinstance(predictions[0], str) if len(predictions) > 0 else False
    
    # Try to get SMILES from inputs
    inputs = getattr(eval_pred, 'inputs', None)
    smiles = []
    
    if inputs is not None:
        # Extract SMILES from inputs (adapt based on your data structure)
        for input_item in inputs:
            if isinstance(input_item, str):
                smiles.append(input_item)
            else:
                smiles.append("placeholder_smile")
    else:
        smiles = ["placeholder_smile"] * len(predictions)
    
    # Process predictions based on type
    processed_predictions = []
    if is_text_output:
        for pred in predictions:
            match = NUMBER_PATTERN.search(pred)
            if match:
                processed_predictions.append(float(match.group(0)))
            else:
                processed_predictions.append(None)
    else:
        processed_predictions = predictions
    
    # Update accumulators with batch data
    compute_metrics_with_batch_support.total_predictions += len(predictions)
    
    for pred, label, smile in zip(processed_predictions, labels, smiles):
        if label is None:
            continue
            
        if pred is not None:
            compute_metrics_with_batch_support.valid_predictions += 1
            
        # Calculate error
        mae = abs(label - pred) if pred is not None else MAX_ERROR
        
        # Group by SMILE
        smile_hash = hash_smile(smile)
        compute_metrics_with_batch_support.smile_groups[smile_hash].append((pred, label, mae))
    
    # If this is the final call, compute summary metrics
    if compute_result:
        all_maes = []
        median_maes = []
        
        for values in compute_metrics_with_batch_support.smile_groups.values():
            valid_answers = [v[0] for v in values if v[0] is not None]
            
            for pred, label, mae in values:
                if pred is not None:
                    all_maes.append(mae)
            
            if valid_answers:
                answer_median = np.median(valid_answers)
                gold = values[0][1]
                mae_median = abs(gold - answer_median)
            else:
                mae_median = MAX_ERROR
                
            median_maes.append(mae_median)
        
        # Calculate final metrics
        metrics = {
            "mae": np.mean(all_maes) if all_maes else MAX_ERROR,
            "median_mae": np.mean(median_maes) if median_maes else MAX_ERROR,
            "valid_predictions_pct": (compute_metrics_with_batch_support.valid_predictions / 
                                     compute_metrics_with_batch_support.total_predictions 
                                     if compute_metrics_with_batch_support.total_predictions > 0 else 0)
        }
        
        # Reset accumulators for next evaluation
        compute_metrics_with_batch_support.smile_groups = defaultdict(list)
        compute_metrics_with_batch_support.valid_predictions = 0
        compute_metrics_with_batch_support.total_predictions = 0
        
        return metrics
    
    # For intermediate batches, return empty dict
    return {}