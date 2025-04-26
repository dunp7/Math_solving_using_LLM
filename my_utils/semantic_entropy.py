"""Semantic Entropy Utilities."""

import torch
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm


def gen_responses_probs(model, tokenizer, question, number_responses=10, temperature=1.0):
    """ Generates 10 responses with high temeperature

    Parameters:
        model (AutoModelForCausalLM): The language generation model
        tokenizer (AutoTokenizer): The tokenizer for the model
        question (str): The input question to generate responses for
        number_responses (int): The number of responses to generate, default 10
        number_responses (float): The number used to modulate the next token probabilities, default 1.0

    Returns:
        dict: The deafult dictionary returned from generate() with token and sequence probabilities
    """

    input_ids = tokenizer(question, return_tensors="pt").to(model.device)
    input_length = input_ids['input_ids'].shape[1]

    outputs_high_temp = model.generate(
        **input_ids,
        max_new_tokens=128,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        top_k=50,                  # top-K sampling
        top_p=0.9,                 # nucleus sampling
        temperature=temperature,
        num_return_sequences=number_responses
    )

    sequence_token_probabilities = []
    generated_answer_tokens = []
    for idx in range(number_responses):
        # Only keep the generated tokens by slicing out the input question tokens
        generated_tokens = outputs_high_temp.sequences[idx, input_length:] 
        generated_answer_tokens.append(generated_tokens.cpu().tolist())
        
        # Calculate probabilities for each token in the generated response
        probabilities = [F.softmax(score, dim=-1) for score in outputs_high_temp.scores] 
        token_probabilities = []
        for i, token_id in enumerate(generated_tokens):
            token_prob = probabilities[i][idx, token_id].item()  # [idx, token_id] for batch dimension
            if token_prob > 0:
                token_probabilities.append(token_prob)
        sequence_token_probabilities.append(token_probabilities)

    outputs = {
        "sequences": generated_answer_tokens,
        # "tokens_probabilities": sequence_token_probabilities,
        "sequences_probabilities": [-np.sum(np.log(prob)) / len(prob) for prob in sequence_token_probabilities], 
    }

    return outputs


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


