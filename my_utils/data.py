"""Data Utilities."""

import datasets
import os
import random
import re


def load_ds(dataset_name, seed):
    """ Loads datasets from Hugging Face
    
    Parameters:
        dataset_name (string): Name of the dataset in the Hugging Face library
        seed (int): The seed for random library

    Returns:
        (Dataset, Dataset): Returns the train and validation datasets

    Parts of function from https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/data/data_utils.py
    """

    train_dataset, validation_dataset = None, None


    if dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {\
            'question': x['question_concat'],
            'type': x['Type'],
            'equation': x['Equation'],
            'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}
        }

        train_dataset = train_dataset.map(reformat)
        validation_dataset = validation_dataset.map(reformat)


    elif dataset_name == 'multiarith':

        train_dataset = datasets.load_dataset("ChilleD/MultiArith", split="train")
        validation_dataset = datasets.load_dataset("ChilleD/MultiArith", split="test")

        reformat = lambda x: {
            'question': x['question'], 'answers' : {'text': [str(x['final_ans'])]}
        }

        train_dataset = train_dataset.map(reformat)
        validation_dataset = validation_dataset.map(reformat)

    elif dataset_name == "gsm8k":
        train_dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
        validation_dataset = datasets.concatenate_datasets([train_dataset, datasets.load_dataset("openai/gsm8k", "main", split="test")])

        def extract_number(answer_text):
            # Use regex to find the first number in the text
            match = re.search(r'####\s*(\d+)', answer_text)
            return match.group(1) if match else None

        reformat = lambda x: {
            'question': x['question'],
            'answers': {'text': [extract_number(str(x['answer']))]},
        }
        train_dataset = train_dataset.map(reformat)
        validation_dataset = validation_dataset.map(reformat)

    else:
        raise ValueError

    return train_dataset, validation_dataset


def sample_ds(dataset, n_samples, seed, name):
    """ Selects random samples from a dataset
    
    Parameters:
        dataset (Dataset): The dataset
        n_samples (int): The number of samples
        seed (int): The seed for random library
        name (string): The name of the dataset

    Returns:
        Dataset: Returns a Dataset with size equal to the n_samples and name is saved in the .info.description
    """

    if n_samples > dataset.num_rows:
        raise ValueError(f"Cannot sample {n_samples} rows from a dataset with only {dataset.num_rows} rows.")

    random.seed(seed)
    random_indices = random.sample(range(dataset.num_rows), n_samples)  # Unique indices
    dataset = dataset.select(random_indices)
    dataset.info.description = name  # Save name in metadata
    print(f"Dataset: {name}")
    print(dataset, "\n")
    return dataset


def load_results(base_path, type):
    """Loads datasets organized by entailment models and their versions from a specified base directory.

    Parameters:
        base_path (str): The root directory containing the datasets. 
                         It is expected to have a structure where each model has its own directory, and within each 
                         model directory are subdirectories for different versions, which contain datasets.
        type (str): The type of entailment being evaluated (e.g., "LLM" for large language models or "Transformer")

    Returns:
        tuple:
            - loaded_datasets (dict): A nested dictionary where the first-level keys are model names, the second-level 
              keys are version names, and the values are lists of dictionaries mapping dataset names to dataset objects.
                Example structure:
                {
                    "model1": {
                        "version1": [{"dataset1": dataset_object1}, {"dataset2": dataset_object2}],
                        "version2": [{"dataset3": dataset_object3}],
                    },
                    "model2": { ... }
                }
            - names (list): A list of strings, where each string represents a model and its size 
                            (e.g., "LLM Model1 Small")
    """
     
    loaded_datasets = {}
    names = []

    # Iterate over the entailment models (directories under base_path)
    for model in os.listdir(base_path):
        model_path = os.path.join(base_path, model)
        if os.path.isdir(model_path):
            loaded_datasets[model] = {}
            
            # Iterate over the versions
            for version in os.listdir(model_path):
                version_path = os.path.join(model_path, version)
                if os.path.isdir(version_path):
                    loaded_datasets[model][version] = []
                    
                    # Iterate over the datasets in each version
                    for dataset_name in os.listdir(version_path):
                        dataset_path = os.path.join(version_path, dataset_name)
                        if os.path.exists(dataset_path):
                            try:
                                dataset_object = datasets.load_from_disk(dataset_path)
                                loaded_datasets[model][version].append({dataset_name: dataset_object})
                                print(f"Loaded {dataset_name} from {model}/{version}")
                            except Exception as e:
                                print(f"Failed to load {dataset_name} from {model}/{version}: {e}")
                    names.append(f"{type} {model.capitalize()} {version}")
                    print()

    return loaded_datasets, names