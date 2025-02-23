import json
import random

def get_random_half_dict(original_dict):
    # Get the number of items to select (half of the dictionary length)
    num_items = len(original_dict) // 3
    
    # Convert dictionary items to a list
    items = list(original_dict.items())
    
    # Randomly select half of the items
    selected_items = random.sample(items, num_items)
    
    # Convert back to dictionary
    result_dict = dict(selected_items)
    
    return result_dict


