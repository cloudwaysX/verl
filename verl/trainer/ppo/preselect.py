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
import pandas as pd
import numpy as np

def balance_dataset_by_ability(df, budget_n, ability_scope=None, random_seed=42):
    """
    Filter a dataframe to include a balanced number of examples across a specified
    subset of ability topics, optionally filtering by a list of allowed abilities first.

    Args:
        df (pd.DataFrame): The input dataframe containing a column 'ability'.
        budget_n (int): The total number of examples to include in the filtered dataframe.
                        This budget applies *after* filtering by ability_scope.
        ability_scope (list, optional): A list of ability names (strings). If provided,
                                        only examples where the 'ability' column's value
                                        is in this list will be considered for balancing.
                                        Defaults to None, meaning all abilities in the
                                        original DataFrame are considered.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A filtered dataframe with approximately balanced examples across
                      the abilities included in the ability_scope (if any), up to budget_n.
    """

    # Ensure reproducibility
    if random_seed is not None:
      np.random.seed(random_seed)

    # --- New: Filter by ability_scope if provided ---
    working_df = df.copy() # Work on a copy to not modify the original dataframe

    if ability_scope is not None:
        if 'ability' not in working_df.columns:
             raise ValueError("DataFrame must contain an 'ability' column.")
        original_len = len(working_df)
        working_df = working_df[working_df['ability'].isin(ability_scope)].copy()
        print(f"Filtered down to {len(working_df)} examples within ability scope: {ability_scope} (from {original_len})")
        if len(working_df) == 0:
            print("No examples remaining after filtering by ability_scope. Returning empty DataFrame.")
            # Return an empty DataFrame with the same columns as the original df
            return pd.DataFrame(columns=df.columns)
    # --- End New ---

    # Count the occurrences of each ability in the (potentially filtered) dataframe
    # Only consider abilities present in the working_df
    ability_counts = working_df['ability'].value_counts()

    print(f"Ability distribution (after ability scope filter if applied):\n{ability_counts}")
    print(f"Number of unique abilities in scope: {len(ability_counts)}")

    # Handle case where budget_n is larger than the available examples after filtering
    if budget_n > len(working_df):
        print(f"Warning: budget_n ({budget_n}) is larger than the available examples within the ability scope ({len(working_df)}). Returning all available examples.")
        return working_df.copy() # Return all filtered examples if budget exceeds available

    # Calculate the ideal number of examples per ability within the filtered set
    num_abilities_in_scope = len(ability_counts)
    if num_abilities_in_scope == 0:
        print("No abilities found in the dataset after filtering by ability_scope. Returning empty DataFrame.")
        return pd.DataFrame(columns=df.columns) # Return empty if no abilities are left

    examples_per_ability = budget_n // num_abilities_in_scope
    remaining = budget_n % num_abilities_in_scope

    print(f"Target: {examples_per_ability} examples per ability, with {remaining} extra examples distributed among abilities in scope")

    # Initialize the list to store selected indices
    selected_indices = []

    # Sort abilities by count (ascending) to allocate extra examples to the least common abilities first within the scope
    # Use ability_counts.index as these are the abilities present in the working_df and within scope
    sorted_abilities_in_scope = ability_counts.sort_values().index.tolist()

    # Allocate the examples for each ability within the scope
    # Iterate through sorted abilities and add extra examples to the first 'remaining' abilities
    for i, ability in enumerate(sorted_abilities_in_scope):
        # Get all indices for this ability from the working_df
        ability_indices = working_df[working_df['ability'] == ability].index.tolist()

        # Determine how many examples to take for this ability
        # Add an extra example for the abilities that come first in the sorted list (least common)
        target_count = examples_per_ability + (1 if i < remaining else 0)

        # If we don't have enough examples for this ability, take all available
        if len(ability_indices) <= target_count:
            selected_indices.extend(ability_indices)
            print(f"Taking all {len(ability_indices)} examples for ability '{ability}' (within scope)")
        else:
            # Randomly sample the required number of examples
            # Ensure we don't sample more than available indices
            sample_count = min(target_count, len(ability_indices))
            sampled_indices = np.random.choice(ability_indices, sample_count, replace=False)
            selected_indices.extend(sampled_indices)
            print(f"Sampled {sample_count} examples for ability '{ability}' (within scope, from {len(ability_indices)} available, target was {target_count})")

    # Create the balanced dataframe using indices from the original dataframe for safety
    # (though indices from working_df should also be valid in the original df)
    balanced_df = df.loc[selected_indices].copy()

    # Print statistics about the balanced dataset
    print("\nBalanced ability distribution (within scope):")
    if not balanced_df.empty:
        print(balanced_df['ability'].value_counts())
    else:
        print("Balanced DataFrame is empty.")
    print(f"Total examples in balanced dataset: {len(balanced_df)}")

    # Verify the total number of examples is close to budget_n, accounting for cases
    # where available examples were less than budget_n
    if len(balanced_df) != budget_n and len(working_df) >= budget_n:
         print(f"Warning: Final dataset size ({len(balanced_df)}) does not match budget_n ({budget_n}). This can happen if some abilities within the scope had fewer examples than the target count per ability.")
    elif len(balanced_df) != len(working_df) and len(working_df) < budget_n:
         print(f"Info: Final dataset size ({len(balanced_df)}) matches the number of available examples within the ability scope ({len(working_df)}) as budget_n was larger.")
    
    return balanced_df



# Sort by length of prompt
def calculate_prompt_length(prompt_list):
    """
    Calculates the length of the 'content' from the first user message
    in a list of prompt dictionaries. Returns 0 for unexpected structures.
    """
    if isinstance(prompt_list, list) and len(prompt_list) > 0:
        first_message = prompt_list[0]
        if isinstance(first_message, dict) and 'content' in first_message:
            return len(str(first_message['content'])) # Ensure it's a string
    elif isinstance(prompt_list, str):
        return len(prompt_list)
    else:
        raise ValueError("Unexpected structure for prompt_list. Expected a list of dictionaries or a string.")
def select_prompts_by_highest_length(df, budget_n):
    """
    Selects a subset of a dataframe containing prompts with the highest lengths,
    up to a specified budget, by calculating length from the 'prompt' column
    (more concise version).

    Args:
        df (pd.DataFrame): The input dataframe containing a column named 'prompt'
                           with a structure like [{"role": "user", "content": "..."}].
        budget_n (int): The maximum number of examples to select.

    Returns:
        pd.DataFrame: A filtered dataframe containing up to budget_n examples
                      with the highest calculated prompt lengths.
                      Returns all examples if budget_n is greater than the
                      number of rows in the dataframe.
        Raises:
            ValueError: If the 'prompt' column does not exist in the dataframe.
    """
    if 'prompt' not in df.columns:
        raise ValueError("The dataframe must contain a 'prompt' column to calculate length.")

    if budget_n is None or budget_n >= len(df):
        print(f"Budget ({budget_n}) is greater than or equal to the total number of examples ({len(df)}). Returning all examples.")
        return df.copy()

    # Calculate lengths and get the indices of the top N using nlargest
    # This is a concise way to find the indices corresponding to the largest values
    # after applying the length calculation.
    top_n_indices = df['prompt'].apply(calculate_prompt_length).nlargest(budget_n).index

    # Select the rows from the original dataframe using the collected indices
    selected_df = df.loc[top_n_indices].copy()

    print(f"Selected {len(selected_df)} examples with the highest calculated prompt lengths (up to budget of {budget_n}).")

    return selected_df
