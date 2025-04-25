"""
P(true) baseline to compare with semantic entropy

based on https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/p_true.py

"""
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
import torch

def calculate_p_true(datasets, entailment_model, entailment_tokenizer, tokenizer, save_path):
    """
    Computes p_true for multiple datasets and saves the updated datasets, handling tokenized answers.

    Parameters:
        datasets (list): List of datasets to process.
        tokenizer (AutoTokenizer): Tokenizer for decoding tokenized answers.
        save_path (str): Directory path to save updated datasets.

    Returns:
        None: Updated datasets are saved to disk.
    """
    for dataset in datasets:
        # Create a deep copy to avoid modifying the original dataset
        dataset_copy = deepcopy(dataset)

        print(f"Processing dataset: {dataset_copy.info.description}")
        p_true_values = []

        # Compute p_true for each row in the dataset
        for row in tqdm(dataset_copy):
            question_concat = row['question_concat']
            generated_answers = row['generated_answers']
            most_deterministic_answer = row['generated_answer_acc']

            # Decode tokenized answers (checking for possible nested structures)
            decoded_generated_answers = []
            for answer in generated_answers:
                # In case 'answer' is a list of sequences, decode each
                decoded_answer = tokenizer.decode(answer['sequences'][0], skip_special_tokens=True) if isinstance(answer, dict) else tokenizer.decode(answer, skip_special_tokens=True)
                decoded_generated_answers.append(decoded_answer)

            # Decode the most deterministic answer
            decoded_most_deterministic_answer = tokenizer.decode(most_deterministic_answer['sequences'][0], skip_special_tokens=True) if isinstance(most_deterministic_answer, dict) else tokenizer.decode(most_deterministic_answer, skip_special_tokens=True)

            # Construct the p_true prompt
            brainstormed_answers_str = ' | '.join(decoded_generated_answers)
            prompt = (
                f"Question: {question_concat}\n"
                f"Brainstormed Answers: {brainstormed_answers_str}\n"
                f"Possible answer: {decoded_most_deterministic_answer}\n"
                "Is the possible answer:\n"
                "A) True\n"
                "B) False\n"
                "The possible answer is:"
            )


            ##### Use only DEberta to compute p_true
            # Compute p_true using the model's method
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).to(entailment_model.device)

            # Forward pass through the DeBERTa model
            with torch.no_grad():
                outputs = entailment_model(**inputs)

            # Get logits and apply softmax to obtain probabilities
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

            # Assuming the true label corresponds to index 0 (True) and false to index 1 (False)
            p_true = probs[0, 0].item()  
            p_true_values.append(p_true)

        # Add the p_true column to the dataset
        dataset_copy = dataset_copy.add_column('p_true', p_true_values)

        # Save the updated dataset to disk
        dataset_path = f"{save_path}/{dataset_copy.info.description}_updated"
        dataset_copy.save_to_disk(dataset_path)
        print(f"Dataset saved to {dataset_path}") 

