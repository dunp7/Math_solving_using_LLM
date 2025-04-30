"""
P(true) baseline to compare with semantic entropy

based on https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/p_true.py

"""
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
import torch


def generate_p_true(datasets, save_path ,entailment_model, entailment_tokenizer, gen_tokenizer):
    """
    Computes p_true for multiple datasets and saves the updated datasets, handling tokenized answers.

    Parameters:
        datasets (list): List of datasets to process.
        gen_tokenizer (AutoTokenizer): Tokenizer for decoding tokenized answers.
        save_path (str): Directory path to save updated datasets.

    Returns:
        None: Updated datasets are saved to disk.
    """
    for dataset in datasets:

        dataset_copy = deepcopy(dataset)

        print(f"Processing dataset: {dataset_copy.info.description}")
        p_true_values = []

         # Compute p_true for each row in the dataset using index
        for idx in tqdm(range(len(dataset_copy)), desc= "Calculating P(True)"):
            question_concat = dataset_copy[idx]['question_concat']
            generated_answers = dataset_copy['generated_answers'][idx]['sequences']

            # Decode the generated answers (list of token sequences)
            decoded_generated_answers = []
            for answer in generated_answers:
                # If answer is a list of token IDs, decode them directly
                if isinstance(answer, list):  
                    decoded_answer = gen_tokenizer.decode(answer, skip_special_tokens=True)
                    decoded_generated_answers.append(decoded_answer)
                else:
                    print("Unexpected format in generated_answers")
                    continue

            # Select the most deterministic answer 
            decoded_most_deterministic_answer = gen_tokenizer.decode(dataset_copy[idx]['generated_answer_acc']['sequences'][0], skip_special_tokens=True)

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

            # Tokenize the prompt
            inputs = entailment_tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).to(entailment_model.device)

            # Forward pass through the DeBERTa model
            with torch.no_grad():
                outputs = entailment_model(**inputs)

            # Get logits and apply softmax to obtain probabilities
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

            # Assuming the true label corresponds to index 0 (True) and false to index 1 (False)
            p_true = probs[0, 0].item()  # Probability of being True
            p_true_values.append(p_true)

        # Add the p_true column to the dataset
        dataset_copy = dataset_copy.add_column('p_true', p_true_values)

        # Save the updated dataset to disk
        dataset_path = f"{save_path}{dataset_copy.info.description}"
        dataset_copy.save_to_disk(dataset_path)
        print(f"Dataset saved to {dataset_path}")

