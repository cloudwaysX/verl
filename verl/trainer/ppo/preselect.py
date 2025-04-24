import numpy as np
import pandas as pd 

def selection_for_math_difficulty(dataframe, lower_bound=3, upper_bound=5):
    assert "difficulty" in dataframe["extra_info"][0], "difficulty is not in the extra_info"
    assert "ability" in dataframe, "ability is not in the data frame"
    # Select the rows with difficulty between the lower and upper bounds
    selected_rows = dataframe.apply(
        lambda x: (not x["ability"].startswith("math")) or (lower_bound<=x["extra_info"]["difficulty"] <=upper_bound),
        axis=1)
    return dataframe[selected_rows]

def selection_for_mathamc_difficulty(dataframe, lower_bound=3, upper_bound=5):
    assert "difficulty" in dataframe["extra_info"][0], "difficulty is not in the extra_info"
    assert "ability" in dataframe, "ability is not in the data frame"
    # Select the rows with difficulty between the lower and upper bounds
    selected_rows1 = dataframe.apply(
        lambda x: (x["ability"].startswith("math")) and (lower_bound<=x["extra_info"]["difficulty"] <=upper_bound),
        axis=1)
    selected_rows2 = dataframe.apply(
        lambda x: (x["ability"].startswith("amc")) and (x["extra_info"]["difficulty"] >= 2),
        axis=1)
    selected_rows = selected_rows1 | selected_rows2
    return dataframe[selected_rows]

def selection_for_deepscaler_difficulty(dataframe, lower_bound=3, upper_bound=8):
    assert "difficulty" in dataframe["extra_info"][0], "difficulty is not in the extra_info"
    # Select the rows with difficulty between the lower and upper bounds
    selected_rows = dataframe.apply(
        lambda x: (x["extra_info"]["difficulty"] is None) or (lower_bound<=x["extra_info"]["difficulty"] <=upper_bound),
        axis=1)
    return dataframe[selected_rows]

def selection_for_openthoughts_difficulty(dataframe, lower_bound=4, upper_bound=4):
    assert "difficulty" in dataframe["extra_info"][0], "difficulty is not in the extra_info"
    # Select the rows with difficulty between the lower and upper bounds
    selected_rows = dataframe.apply(
        lambda x: (x["extra_info"]["difficulty"] is None) or (lower_bound<=x["extra_info"]["difficulty"] <=upper_bound),
        axis=1)
    return dataframe[selected_rows]

# Add balance function
def balance_dataset_by_ability(df, budget_n, random_seed=42):
    """
    Filter a dataframe to include a balanced number of examples across different ability topics.
    
    Args:
        df (pd.DataFrame): The input dataframe containing a column 'ability'
        budget_n (int): The total number of examples to include in the filtered dataframe
        random_seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: A filtered dataframe with approximately balanced examples across ability topics
    """
    
    # Ensure reproducibility
    if random_seed is not None:
      np.random.seed(random_seed)
    
    # Count the occurrences of each ability
    ability_counts = df['ability'].value_counts()
    print(f"Original ability distribution:\n{ability_counts}")
    print(f"Number of unique abilities: {len(ability_counts)}")
    
    # Calculate the ideal number of examples per ability
    examples_per_ability = budget_n // len(ability_counts)
    remaining = budget_n % len(ability_counts)
    
    print(f"Target: {examples_per_ability} examples per ability, with {remaining} extra examples distributed")
    
    # Initialize the list to store selected indices
    selected_indices = []
    
    # Sort abilities by count to allocate extra examples to the least common abilities
    sorted_abilities = ability_counts.sort_values().index.tolist()
    
    # Allocate the examples for each ability
    for i, ability in enumerate(sorted_abilities):
        # Get all indices for this ability
        ability_indices = df[df['ability'] == ability].index.tolist()
        
        # Determine how many examples to take for this ability
        # Add an extra example for the least common abilities if there are remaining examples
        target_count = examples_per_ability + (1 if i < remaining else 0)
        
        # If we don't have enough examples for this ability, take all available
        if len(ability_indices) <= target_count:
            selected_indices.extend(ability_indices)
            print(f"Taking all {len(ability_indices)} examples for ability '{ability}'")
        else:
            # Randomly sample the required number of examples
            sampled_indices = np.random.choice(ability_indices, target_count, replace=False)
            selected_indices.extend(sampled_indices)
            print(f"Sampled {target_count} examples for ability '{ability}' (from {len(ability_indices)} available)")
    
    # Create the balanced dataframe
    balanced_df = df.loc[selected_indices].copy()
    
    # Print statistics about the balanced dataset
    print("\nBalanced ability distribution:")
    print(balanced_df['ability'].value_counts())
    print(f"Total examples in balanced dataset: {len(balanced_df)}")
    
    return balanced_df