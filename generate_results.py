"""Functions for generating and analyzing responses"""

from torch.cuda import empty_cache
from tqdm import tqdm
from copy import deepcopy
from my_utils.semantic_entropy import gen_responses_probs, cluster_responses, calculate_sem_entr
from my_utils.metrics import assess_acc_llm, assess_acc_SQuAD, assess_acc_gemini
import time

def generate_answers(datasets, data_answers_path, llm_model, llm_tokenizer,acc_model= None, acc_tokenizer = None ,intro_promt= None, acc_flg=0, api_key = None):
    """Generates responses and accuracy labels for questions in multiple datasets using a specified language model

    Parameters:
        datasets (list): A list of datasets, where each dataset contains questions and ground-truth answers
        data_answers_path (str): The directory path where the updated datasets with generated answers will be saved
        llm_model (AutoModelForCausalLM): The language model used to generate responses
        llm_tokenizer (AutoTokenizer): The tokenizer associated with the language model
        intro_promt (str, optional): An introductory prompt for the language model. Defaults to None.
        acc_flg (int, optional): 2 for Gemini, 1 for LLM, 0 for SQuAD. Defaults to 0.
    Returns:
        None: The results are directly saved to disk
    """

    

    if not intro_promt: 
        intro_promt = "Answer the following question in a single brief but complete sentence. "
    print(intro_promt)
    for dataset in datasets:
        all_responses = []
        all_acc_resp = []
        all_labels = []

        print(f"\nGenerating responses for {dataset.info.description} dataset...")

        for i in tqdm(range(len(dataset))):
            # Generate responses for Semantic Entropy and Accuracy
            prompt = intro_promt + dataset[i]["question"]
            responses = gen_responses_probs(llm_model, llm_tokenizer, prompt)
            empty_cache()
            acc_response = gen_responses_probs(llm_model, llm_tokenizer, prompt, number_responses=1, temperature=0.1)
            empty_cache()
            acc_response_text = llm_tokenizer.decode(acc_response["sequences"][0], skip_special_tokens=True)
            empty_cache()
            if acc_flg == 0:
                label = assess_acc_SQuAD(str(dataset[i]["answers"]["text"]), acc_response_text)
            elif acc_flg == 1:
                label = assess_acc_llm(acc_model, acc_tokenizer, dataset[i]["question"], str(dataset[i]["answers"]["text"]), acc_response_text)
            elif acc_flg == 2:
                label = assess_acc_gemini(api_key, dataset[i]["question"], dataset[i]["answers"]["text"], acc_response_text)
                time.sleep(3)
            empty_cache()
            all_responses.append(responses)
            all_acc_resp.append(acc_response)
            all_labels.append(label)
            
        # Save results to dataset
        dataset = dataset.add_column("generated_answers", all_responses)
        dataset = dataset.add_column("generated_answer_acc", all_acc_resp)
        dataset = dataset.add_column("labels", all_labels)
        dataset.save_to_disk(data_answers_path + dataset.info.description)  


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