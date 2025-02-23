def validate_dataset(dataset):
    """Perform basic validation checks on the dataset."""
    
    # Define the required fields for the dataset
    required_fields = ["problem", "prompt"]

    # Loop through the 'train' and 'test' splits of the dataset
    for split in ['train', 'test']:
        print(f"\nValidating {split} split:")

        # Retrieve column names from the dataset
        fields = dataset[split].column_names

        # Check if any required fields are missing
        missing = [field for field in required_fields if field not in fields]
        if missing:
            print(f"Warning: Missing fields: {missing}")  # Warn if fields are missing
        else:
            print("✓ All required fields present")  # Confirm all fields are present

        # Retrieve the first sample from the dataset split
        sample = dataset[split][0]

        # Extract the 'prompt' field, which contains a list of messages
        messages = sample['prompt']

        # Validate the prompt format:
        # - It should contain at least two messages
        # - The first message should be from the 'system' role
        # - The second message should be from the 'user' role
        if (len(messages) >= 2 and
            messages[0]['role'] == 'system' and
            messages[1]['role'] == 'user'):
            print("✓ Prompt format is correct")  # Confirm correct format
        else:
            print("Warning: Incorrect prompt format")  # Warn if format is incorrect