"""
Generate Predictive Entropy (PE) and Length_normalized PE (Ln-PE)

"""
import tqdm
import numpy as np
from copy import deepcopy


def calculate_PE(dataset, length_normalized=False):
    """
    Calculate the Predictive Entropy (PE) for a given dataset using already provided sequence probabilities.

    Parameters:
        dataset (list): A list of datasets, where each dataset contains sequence probabilities for multiple generated sequences.
        length_normalized (bool): Whether to normalize the PE by sequence length (optional).
    
    Returns:
        list: A list of PE values for each example in the dataset.
    """
    pe_values = []

    for i in tqdm(range(len(dataset)), desc="Calculating PE"):
        sequence_token_probs = dataset[i]["generated_answers"]["sequences_probabilities"]
        pe_value = 0
        
        # Calculate PE as -sum(p log(p)) for each token in the sequence
        for prob in sequence_token_probs:
            pe_value += -prob * np.log(prob) if prob > 0 else 0  


        if length_normalized == True:
            pe_value /= len(sequence_token_probs)

        pe_values.append(pe_value)

    return pe_values


def generate_PE(datasets, save_path):
    """
    Generate both PE (Predictive Entropy) and LN_PE (Length Normalized Predictive Entropy) for multiple datasets.

    Parameters:
        datasets (list): A list of datasets (each dataset contains sequence probabilities for multiple examples).
    
    Returns:
        dict: A dictionary with dataset names as keys and a tuple of PE and LN_PE values as values.
    """


    for dataset in datasets:
        # Create a deep copy to avoid modifying the original dataset
        dataset_copy = deepcopy(dataset)

        print(f"Processing dataset: {dataset_copy.info.description}") 

        # Calculate PE and LN_PE for the current dataset
        pe_values = calculate_PE(dataset_copy, length_normalized=False)  # PE 
        ln_pe_values = calculate_PE(dataset_copy, length_normalized=True)  # LN_PE 


        # Add 2 PE measures column to the dataset
        dataset_copy = dataset_copy.add_column('PE', pe_values)
        dataset_copy = dataset_copy.add_column('Ln-PE', ln_pe_values)


        # Save the updated dataset to disk
        dataset_path = f"{save_path}{dataset_copy.info.description}"
        dataset_copy.save_to_disk(dataset_path)
        print(f"Dataset saved to {dataset_path}")

