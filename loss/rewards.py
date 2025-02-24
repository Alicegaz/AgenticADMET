# Import necessary libraries
import re

# Import math-related utilities
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import numpy as np
#source openr1/bin/activate



def accuracy_reward(completions, ground_truth=None, log_normalize=False, **kwargs):
    """
    Reward function to check if the model's response is mathematically 
    equivalent to the ground truth solution.
    Uses latex2sympy2 for parsing and math_verify for validation.
    """
    # print("completions", completions)
    #TODO: reward right niw is -inf;0, 0 for perfect match between groundtruth and prediciton, howether when solution is None (now we exluded such cases) we set reward to 0.5, which violates our current range
    #TODO: does the current range of -inf;0 
    # print(ground_truth, kwargs.keys())

    # Extract responses
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    solutions = kwargs.get("solution") # Get solutions from kwargs
    property = kwargs.get("property")

    if solutions is None:
        return [0.5] * len(completions) # Return neutral reward if no solution
    
    for content, gold_val in zip(contents, solutions):
        # Parse the ground truth solution
        # gold_parsed = parse(sol, extraction_mode="first_match", 
        #                     extraction_config=[LatexExtractionConfig()])
        
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
                extraction_mode="first_match",
            )

            # Reward 1.0 if correct, 0.0 if incorrect
            # reward = float(verify(answer_parsed, gold_parsed))
            try:
                # Evaluate symbolic expressions as floats
                # gold_val = float(gold_parsed.evalf())
                # answer_val = float(answer_parsed.evalf())
                answer_val = float(answer_parsed[0])
                if log_normalize and property != "LogD":
                    answer_val = np.log(answer_val+1)
                    gold_val = np.log(gold_val+1)
                reward = -1*np.mean(np.abs(gold_val - answer_val))
                reward = np.clip(1-(1/6)*reward, 0, 1)
                print("parsed correctly", answer_val, gold_val)
            except Exception as e:
                # If direct numeric eval fails 
                # (e.g., the expression is more complicated),
                # we can fallback to 'verify' for symbolic equivalence:
                # reward = float(verify(answer_parsed, gold_val))
                # if reward == 0: #gives vero for not equal
                #     print("answer parsed 0", answer_parsed)
                reward = 0
                print(e, answer_parsed[0])
        else:
            # If ground truth cannot be parsed, assign neutral reward (0.5)
            reward = 0.5
            print("Warning: Gold solution is None:", gold_val)

        rewards.append(reward)
    
    return rewards


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
def get_reward_functions(script_args):
    """
    Returns a list of reward functions based on the script arguments.
    """
    reward_funcs_list = []
    reward_funcs_registry = {
        "accuracy": accuracy_reward,  # Assuming accuracy_reward is defined in previous steps
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
    }

    for func_name in script_args.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list