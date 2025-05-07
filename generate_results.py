"""Functions for generating and analyzing responses"""

from torch.cuda import empty_cache
from tqdm import tqdm
from copy import deepcopy
from my_utils.metrics import assess_acc_llm, assess_acc_SQuAD, assess_acc_gemini
import time
import torch.nn.functional as F
import numpy as np


def generate_labels(datasets, gen_tokenizer, save_path ,acc_model=None, acc_tokenizer=None, api_key=None):
    """
    Determines the label for the accuracy of the generated response.

    Parameters:
        datasets (dict): A single dataset item containing the question and ground-truth answer
        acc_response_text (str): The generated response for accuracy evaluation
        acc_model (Optional): The accuracy evaluation model (used if acc_flg is 1)
        acc_tokenizer (Optional): The tokenizer associated with the accuracy evaluation model (used if acc_flg is 1)
        api_key (Optional): API key for Gemini (used if acc_flg is 2)

    Returns:
        label (Any): The evaluated label for the response accuracy
    """
    for dataset in datasets:
        labels_SQuAD = []
        labels_LLM = []
        labels_Gemini = []
        dataset_copy = deepcopy(dataset)

        for i in tqdm(range(len(dataset_copy)), desc="Generating Label"):

            acc_response_text = gen_tokenizer.decode(dataset_copy['generated_answer_acc'][i]["sequences"][0], skip_special_tokens=True)
            empty_cache()
            # F1 from SQuAD
            label_SQuAD = assess_acc_SQuAD(str(dataset_copy[i]["answers"]["text"]), acc_response_text)

            # LLM
            label_LLM = assess_acc_llm(acc_model, acc_tokenizer, dataset_copy[i]["question"], str(dataset_copy[i]["answers"]["text"]), acc_response_text)

            # Gemini API
            label_Gemini = assess_acc_gemini(api_key, dataset_copy[i]["question"], str(dataset_copy[i]["answers"]["text"]), acc_response_text)
            time.sleep(3)

            empty_cache()
            labels_SQuAD.append(label_SQuAD)
            labels_LLM.append(label_LLM)
            labels_Gemini.append(label_Gemini)


        # Save results to dataset
        dataset = dataset.add_column("labels_SQuAD", labels_SQuAD)
        dataset = dataset.add_column("labels_LLM", labels_LLM)
        dataset = dataset.add_column("labels_Gemini", labels_Gemini)
        dataset.save_to_disk(save_path+ dataset.info.description)  



def generate_answers(datasets, data_answers_path, llm_model, llm_tokenizer, instruct_prompt= None):
    """Generates responses and accuracy labels for questions in multiple datasets using a specified language model

    Parameters:
        datasets (list): A list of datasets, where each dataset contains questions and ground-truth answers
        data_answers_path (str): The directory path where the updated datasets with generated answers will be saved
        llm_model (AutoModelForCausalLM): The language model used to generate responses
        llm_tokenizer (AutoTokenizer): The tokenizer associated with the language model
        intro_promt (str, optional): An introductory prompt for the language model. Defaults to None.
    Returns:
        None: The results are directly saved to disk
    """

    

    if not instruct_prompt: 
        instruct_prompt = ""
    for dataset in datasets:
        all_responses = []
        all_acc_resp = []

        print(f"\nGenerating responses for {dataset.info.description} dataset...")

        for i in tqdm(range(len(dataset)), desc= "Generating Answers"):
            # Generate responses for Semantic Entropy and Accuracy
            prompt = f"""
            {instruct_prompt}.\n Answer the following question in a single brief but complete sentence.
            \n Question: {dataset[i]["question"]}
            \n Answer:
            """

            responses = gen_responses_probs(llm_model, llm_tokenizer, prompt)
            empty_cache()
            acc_response = gen_responses_probs(llm_model, llm_tokenizer, prompt, number_responses=1, temperature=0.1)
            empty_cache()
            all_responses.append(responses)
            all_acc_resp.append(acc_response)
      
        # Save results to dataset
        dataset = dataset.add_column("generated_answers", all_responses)
        dataset = dataset.add_column("generated_answer_acc", all_acc_resp)
        dataset.save_to_disk(data_answers_path + dataset.info.description)  


def gen_responses_probs(model, tokenizer, question, number_responses=10, temperature=1.0):
    """ Generates 10 responses with high temeperature

    Parameters:
        model (AutoModelForCausalLM): The language generation model
        tokenizer (AutoTokenizer): The tokenizer for the model
        question (str): The input question to generate responses for
        number_responses (int): The number of responses to generate, default 10

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