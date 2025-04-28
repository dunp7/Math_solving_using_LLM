"""Semantic Entropy Utilities."""

import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.cuda import empty_cache


def generate_SE(datasets, data_entail_path, llm_tokenizer, entail_model, entail_tokenizer, entail_function):
    """Computes Semantic Entropy (SE) and clusters responses for questions in multiple datasets

    Parameters:
        datasets (list): A list of datasets, where each dataset contains questions and previously generated answers
        data_entail_path (str): The directory path where the updated datasets with SE and clusters will be saved
        llm_tokenizer (AutoTokenizer): The tokenizer for decoding responses
        entail_model (AutoModelForSequenceClassification or AutoModelForCausalLM): The model used for entailment evaluation
        entail_tokenizer (AutoTokenizer): The tokenizer associated with the entailment model
        entail_function (function): Fuction that will be used to assess bidirectional entailment

    Returns:
        None: The results are directly saved to disk
    """

    for dataset in datasets:
        all_clusters = []
        all_sem_entr = []
        all_mem_alloc = []
        dataset_copy = deepcopy(dataset)

        print(f"\nGenerating Semantic Entropies for {dataset_copy.info.description} dataset...")

        for i in tqdm(range(len(dataset_copy))):
            # Calculate semantic entropy
            clusters, memory = cluster_responses(dataset_copy[i]["generated_answers"], llm_tokenizer, entail_function, 
                                                 entail_model, entail_tokenizer, question=dataset_copy[i]["question"])
            empty_cache()
            sem_entr = calculate_sem_entr(clusters, dataset_copy[i]["generated_answers"]["sequences_probabilities"])
            all_clusters.append(clusters)
            all_sem_entr.append(sem_entr)
            all_mem_alloc.append(memory)

        # Save results to dataset
        dataset_copy = dataset_copy.add_column("clusters", all_clusters)
        dataset_copy = dataset_copy.add_column("semantic_entropy", all_sem_entr)
        dataset_copy = dataset_copy.add_column("memory_allocation", all_mem_alloc)
        dataset_copy.save_to_disk(data_entail_path + dataset_copy.info.description) 




def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on token embeddings while considering the attention mask.
    This ensures that padding tokens do not contribute to the resulting sentence embedding.

    Parameters:
        model_output (torch.Tensor): The output from the model's forward pass and the first element contains the token embeddings
        attention_mask (torch.Tensor): A tensor of shape [batch_size, seq_length] indicating the attention mask

    Returns:
        torch.Tensor: A tensor of shape [batch_size, embedding_dim] representing the pooled sentence embeddings for each input sentence.
    """

    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def is_entailment_transformer(model, tokenizer, premise, hypothesis, question=""):
    """ Checks if two sentences have bidirectional entailment using a transformer-based model
    
    Parameters:
        model (AutoModelForSequenceClassification): The Sequence classification model
        tokenizer (AutoTokenizer): The tokenizer for the Sequence classification model
        premise (string): The first sentence to be evaluated
        hypothesis (string): The second sentence to be evaluated for entailment
        question (string): The question context related to the sentences (not used in this function, only for compatibility)

    Returns:
        tuple: (Returns True if both sentences entail each other (bidirectional entailment) otherwise False, memory allocation)
    """

    mem_before = torch.cuda.memory_allocated()

    # premise -> hypothesis
    input_ids = tokenizer(premise, hypothesis, return_tensors="pt").to(model.device)
    outputs = model(**input_ids)
    logits = outputs.logits # Get the entailment scores
    entailment_prob = torch.softmax(logits, dim=1)
    label_pre_hypo = torch.argmax(entailment_prob, dim=1).item()

    # hypothesis -> premise
    input_ids = tokenizer(hypothesis, premise, return_tensors="pt").to(model.device)
    outputs = model(**input_ids)
    logits = outputs.logits # Get the entailment scores
    entailment_prob = torch.softmax(logits, dim=1)
    label_hypo_pre = torch.argmax(entailment_prob, dim=1).item()

    mem_after = torch.cuda.memory_allocated()

    return ((label_pre_hypo == 2) and (label_hypo_pre == 2)), mem_after-mem_before

def cluster_responses(responses, llm_tokenizer, is_entailment, entail_model, entail_tokenizer, question):
    """ Create the clusters from the responses
    
    Parameters:
        responses (list): The Sequence classification model
        llm_tokenizer (AutoTokenizer): The tokenizer for the LLM model
        is_entailment (function): The function that will be used to asses the enatilment
        entail_model (AutoModelForCausalLM): The Sequence classification model
        entail_tokenizer (AutoTokenizer): The tokenizer for the Sequence classification model
        question (string): A question providing context for assessing the responses (if it is applicable)

    Returns:
        tuple: (A list where each cluster is represented by another list with the index of the responses,
                total memory allocation for the clustering process)
    """

    clusters = [[0]]
    total_memory = 0
    
    for i in range(1, len(responses['sequences'])):
        for c in clusters:
            response_text = llm_tokenizer.decode(responses['sequences'][i], skip_special_tokens=True)
            cluster_text = llm_tokenizer.decode(responses['sequences'][c[0]], skip_special_tokens=True)            
            entails, memory = is_entailment(entail_model, entail_tokenizer, response_text, cluster_text, question=question)
            total_memory += memory

            if entails:
                c.append(i)
                break
            else:
                clusters.append([i])
                break  
    
    return clusters, total_memory


def calculate_sem_entr(clusters, sequences_prob):
    """ Calculates Semantic Entropy from clustered responses

    Parameters:
        clusters (List): A list where each cluster is represented by another list with the index of the responses
        sequences_prob (List): A list with the sequence probability of each response

    Returns:
        float: The Semantic Entropy of all clusters
    """

    sem_entr = 0.0
    norm_prob = sum(sequences_prob)
    
    for cluster in clusters:
        cluster_prob = sum(sequences_prob[i] for i in cluster) / norm_prob
        sem_entr += cluster_prob * np.log(cluster_prob)
    
    return -sem_entr


