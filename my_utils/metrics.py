"""Calculate metrics"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def assess_acc(model, tokenizer, question, answers, response):
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
              f"The expected answer is: {answers}\n"
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
        auroc = roc_auc_score(1-np.array(d["labels"]), d["semantic_entropy"])
        auroc_list.append(auroc)
        print(f"{d.info.description:20} dataset: {auroc:8.4f}")

    return auroc_list


def calculate_rejection_accuracies(confidences, predictions, true_labels):
    """Computes rejection accuracies for a range of rejection percentages.

    Parameters:
        confidences (array-like): A list or array of confidence scores associated with predictions
        predictions (array-like): A list or array of predicted labels 
        true_labels (array-like): A list or array of true labels corresponding to the predictions

    Returns:
        np.array: An array of accuracies corresponding to rejection percentages of 0%, 10%, ..., 100%. Each value 
                  represents the accuracy of the remaining data after rejecting the specified percentage of least 
                  confident predictions.
    """

    sorted_indices = np.argsort(-confidences)  # Sort by confidence descending
    sorted_predictions = predictions[sorted_indices]
    sorted_true_labels = true_labels[sorted_indices]
    
    accuracies = []
    for reject_percent in np.linspace(0, 1, 11):  # Rejection percentages: 0%, 10%, ..., 100%
        keep_count = int((1 - reject_percent) * len(confidences))
        if keep_count > 0:
            accuracy = np.mean(sorted_predictions[:keep_count] == sorted_true_labels[:keep_count])
        else:
            accuracy = 0  # No data left to compute accuracy
        accuracies.append(accuracy)
    return np.array(accuracies)


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
        rej_acc = calculate_rejection_accuracies(
            np.array(d["semantic_entropy"]),
            np.where(np.array(d["semantic_entropy"]) >= 0.5, 1, 0), 
            1-np.array(d["labels"])
        )
        rejection_percentages = np.linspace(0, 100, 11)  # Rejection percentages in %
        rej_acc_list.append(rej_acc[1:-1])
        aurac = np.trapz(rej_acc, rejection_percentages) / 100
        aurac_list.append(aurac)
        print(f"{d.info.description:20} dataset: {aurac:8.4f}")

    return (aurac_list, rej_acc_list)


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
            elif metric == "AURAC %":
                print(f"\nAURAC % scores for {model.capitalize()} {size}")
                result = calculate_aurac(only_datasets)[1]
            elif metric == "F1":
                print(f"\nF1 scores for {model.capitalize()} {size}")
                result = calculate_f1(only_datasets)
            elif metric == "SE":
                results += [dataset["semantic_entropy"] for dataset in only_datasets]
                continue
            elif metric == "MEMORY":
                print(f"\nMemory allocation in MB for {model.capitalize()} {size}")
                result = calculate_mem_mean_std(only_datasets)
            else:
                print(f"Please specify one of the following Metrics: 'AUROC', 'AURAC', 'AURAC %', 'F1', 'SE', 'MEMORY'")
                return
            
            results.append(result)
    
    return results