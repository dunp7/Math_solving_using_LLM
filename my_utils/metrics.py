"""Calculate metrics"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn import metrics
from torchmetrics.text import SQuAD 


def assess_acc_SQuAD(response, answer):
    """Assesses the semantic equivalence between a proposed response and the expected answer for a given question
    """
    answer = [{"answers": {"text": [answer]}, "id": "1"}] # Target
    response = [{"prediction_text": response, "id": "1"}] # Prediction
    

    squad_metric = SQuAD()
    metrics = squad_metric(response, answer)
    f1_scores = metrics['f1']
 
    return 1 if f1_scores > 0.5 else 0


def assess_acc_llm(model, tokenizer, question, answer, response):
    """Assesses the semantic equivalence between a proposed response and the expected answer for a given question

    Parameters:
        model (AutoModelForCausalLM): A language model used to evaluate the response
        tokenizer (AutoTokenizer): The tokenizer associated with the language model
        question (str): The question to which the answers are related
        answers (str): The ground-truth answer(s) to the question
        response (str): The proposed answer to be assessed

    Returns:
        int: Returns 1 if the model determines the response is equivalent to the expected answer, otherwise 0
    """

    prompt = (f"We are assessing the quality of answers to the following question: {question}\n"
              f"The expected answer is: {answer}\n"
              f"The proposed answer is: {response}\n"
              f"Within the context of the question, does the proposed answer mean the same as the expected answer?\n"
              f"Respond only with yes or no.")

    acc_input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = acc_input_ids["input_ids"].shape[1]
    output = model.generate(**acc_input_ids, max_new_tokens=256, return_dict_in_generate=True)
    text_res = tokenizer.decode(output.sequences[0, input_length:], skip_special_tokens=True).lower()
    
    
    return 1 if "yes" in text_res else 0


def calculate_auroc(datasets):
    """Computes the AUROC for each dataset

    Parameters:
        datasets (list): A list of datasets, where each dataset contains labels (binary ground-truth) and semantic 
                         entropy values for the responses

    Returns:
        list: A list of AUROC scores, one for each dataset
    """

    auroc_list = []
    for d in datasets:
        y_true = np.array(d["labels"])  # No inversion
        y_score = np.array(d["semantic_entropy"])
        
        # Compute ROC curve
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        auroc = metrics.auc(fpr, tpr)

        auroc_list.append(auroc)
        print(f"{d.info.description:20} dataset: {auroc:8.4f}")

    return auroc_list


def accuracy_at_quantile(accuracies, uncertainties, quantile):
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    return np.mean(accuracies[select])


def calculate_aurac(datasets):
    """Computes the AURAC for each dataset

    Parameters:
        datasets (list): A list of datasets, where each dataset contains labels (binary ground-truth) and semantic 
                         entropy values for the responses

    Returns:
        list: A list of AURAC scores, one for each dataset
    """

    aurac_list = []
    rej_acc_list = []

    for d in datasets:
        
        semantic_entropy = np.array(d["semantic_entropy"])
        labels = np.array(d["labels"])

        rejection_percentages = np.linspace(0.1, 1, 20)  
        rej_acc = [
            accuracy_at_quantile(labels, semantic_entropy, q)
            for q in rejection_percentages
        ]
        rej_acc_list.append(rej_acc)


        dx = rejection_percentages[1] - rejection_percentages[0]
        aurac = np.sum(np.array(rej_acc) * dx)  
        aurac_list.append(aurac)

        print(f"{d.info.description:20} dataset: {aurac:8.4f}")

    return aurac_list, rej_acc_list



def calculate_f1(datasets):
    """Computes the F1 scores for each dataset

    Parameters:
        datasets (list): A list of datasets, where each dataset contains labels (binary ground-truth) and semantic 
                         entropy values for the responses

    Returns:
        list: A list of F1 scores, one for each dataset
    """

    f1_list = []

    for d in datasets:
        y_true = 1-np.array(d["labels"])
        y_pred = np.where(np.array(d["semantic_entropy"]) >= 0.5, 1, 0)
        f1 = f1_score(y_true, y_pred)
        f1_list.append(f1)
        print(f"{d.info.description:20} dataset: {f1:8.4f}")

    return f1_list


def calculate_mem_mean_std(datasets):
    """Computes mean and standard deviation (std) of memmeory allocation for each dataset

    Parameters:
        datasets (list): A list of datasets, where each dataset contains a list with memeory allocation  
                         during clustering

    Returns:
        tuple: (A list with the memoery means, A list with the memoery stds) one for each dataset
    """

    mem_means = []
    mem_stds =[]

    for d in datasets:
        d_MB = np.array(d["memory_allocation"])/1e6 # Bytes -> MB
        mean = np.mean(d_MB)
        std = np.std(d_MB)
        print(f"{d.info.description:10} |  Mean: {mean:8.3f}    Std: {std:7.3f}")
        mem_means.append(mean)
        mem_stds.append(std)

    return (mem_means, mem_stds)


def metric_entail_models(model_results, metric):
    """Computes various performance metrics or other properties for different entailment models and their respective 
       sizes across datasets.
    
    Parameters:
        model_results (dict): A nested dictionary containing results for various models and their sizes
                              Structure example:
                              {
                                  "model1": {
                                      "0.5B": [{dataset_name: dataset_object}, ...],
                                      "3.0B": [{dataset_name: dataset_object}, ...],
                                  },
                                  "model2": {
                                      "14.0B": [{dataset_name: dataset_object}, ...],
                                      "30.0B": [{dataset_name: dataset_object}, ...],
                                  },
                                  ...
                              }
        metric (str): The metric/propertie that will be calculated ('AUROC', 'AURAC', 'AURAC %', 'F1', 'SE', 'MEMORY')

    Returns:
        results (list): A list with the results from the selected metric/propertie  for each model and size combination,
                        computed across datasets
    """

    results = []

    for model in model_results.keys():
        for size in model_results[model].keys():
            only_datasets = [list(item.values())[0] for item in model_results[model][size]]

            if metric == "AUROC":
                print(f"\nAUROC scores for {model.capitalize()} {size}")
                result = calculate_auroc(only_datasets)
            elif metric == "AURAC":
                print(f"\nAURAC scores for {model.capitalize()} {size}")
                result = calculate_aurac(only_datasets)[0]
            elif metric == "SE":
                results += [dataset["semantic_entropy"] for dataset in only_datasets]
                continue
            elif metric == "MEMORY":
                print(f"\nMemory allocation in MB for {model.capitalize()} {size}")
                result = calculate_mem_mean_std(only_datasets)
            else:
                print(f"Please specify one of the following Metrics: 'AUROC', 'AURAC', 'SE', 'MEMORY'")
                return
            
            results.append(result)
    
    return results