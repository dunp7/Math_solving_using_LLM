"""Calculate metrics"""

import numpy as np
from sklearn import metrics
from torchmetrics.text import SQuAD 
from google import genai
import os

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

def assess_acc_gemini(api_key, question, answer, response):
    """Assesses the semantic equivalence between a proposed response and the expected answer for a given question

    Parameters:
        api_key (str): The API key for the Gemini API
        question (str): The question to which the answers are related
        answers (str): The ground-truth answer(s) to the question
        response (str): The proposed answer to be assessed

    Returns:
        int: Returns 1 if the model determines the response is equivalent to the expected answer, otherwise 0
    """
    client = genai.Client(api_key= api_key)
    prompt = (f"We are assessing the quality of answers to the following question: {question}\n"
            f"The expected answer is: {answer}\n"
            f"The proposed answer is: {response}\n"
            f"Within the context of the question, does the proposed answer mean the same as the expected answer?\n"
            f"Respond only with yes or no.")

    # Use Gemini API for text generation
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    if "yes" in response.text.lower():
        return 1
    else:
        return 0


def calculate_auroc(datasets, measure_type='SE', save_path="results/", label_type="SQuAD"):
    """
    Computes the AUROC for each dataset based on the chosen label type or majority voting.

    Parameters:
        datasets (list): A list of datasets, where each dataset contains labels (binary ground-truth) and semantic
                         entropy values for the responses.
        measure_type (str): The type of measure to use for scores ('SE', 'p_true', 'PE', 'Ln-PE').
        save_path (str): The path to save the AUROC results.
        label_type (str): The type of label to use for y_true ('SQuAD', 'LLM', 'Gemini', or 'majority').
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)  

    if not os.path.exists(save_path + "AUROC"):
        with open(save_path, "w") as f:
            pass  # Create an empty file

    auroc_list = []
    results = []

    for d in datasets:
        # Select labels based on label_type
        if label_type == "SQuAD":
            y_true = np.array(d["labels_SQuAD"])
        elif label_type == "LLM":
            y_true = np.array(d["labels_LLM"])
        elif label_type == "Gemini":
            y_true = np.array(d["labels_Gemini"])
        elif label_type == "majority":
            # Majority vote logic (consider 1 if more than 2 labels are 1, otherwise 0)
            y_true = np.array([
                1 if sum([d["labels_SQuAD"][i], d["labels_LLM"][i], d["labels_Gemini"][i]]) >= 2 else 0
                for i in range(len(d["labels_SQuAD"]))
            ])
        else:
            raise ValueError(f"Invalid label_type: {label_type}. Choose 'SQuAD', 'LLM', 'Gemini', or 'majority'.")

        # Select measure scores
        if measure_type == 'SE':
            y_score = np.array(d["semantic_entropy"])
        elif measure_type == 'p_true':
            y_score = np.array(d["p_true"])
        elif measure_type == 'PE':
            y_score = np.array(d["PE"])
        elif measure_type == 'Ln-PE':
            y_score = np.array(d["Ln-PE"])
        else:
            raise ValueError(f"Invalid measure_type: {measure_type}. Choose 'SE', 'p_true', 'PE', or 'Ln-PE'.")

        # Compute ROC curve and AUROC
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        auroc = metrics.auc(fpr, tpr)

        auroc_list.append(auroc)
        result = f"{d.info.description}-{measure_type:20} {label_type:10} dataset: {auroc:8.4f}"
        print(result)
        results.append(result)

    # Save results to a .txt file
    with open(save_path, "a") as f:
        f.write('\n')
        f.write("\n".join(results))




def accuracy_at_quantile(accuracies, uncertainties, quantile):
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    return np.mean(accuracies[select])


def calculate_aurac(datasets, measure_type = 'SE', save_path = "results/", label_type = 'SQuAD'):
    """Computes the AURAC for each dataset

    Parameters:
        datasets (list): A list of datasets, where each dataset contains labels (binary ground-truth) and semantic 
                         entropy values for the responses

    Returns:
        list: A list of AURAC scores, one for each dataset
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)  

    if not os.path.exists(save_path + "AURAC"):
        with open(save_path, "w") as f:
            pass  # Create an empty file
    aurac_list = []
    rej_acc_list = []
    results = []
    for d in datasets:

        # Select labels based on label_type
        if label_type == "SQuAD":
            labels = np.array(d["labels_SQuAD"])
        elif label_type == "LLM":
            labels = np.array(d["labels_LLM"])
        elif label_type == "Gemini":
            labels = np.array(d["labels_Gemini"])
        elif label_type == "majority":
            # Majority vote logic (consider 1 if more than 2 labels are 1, otherwise 0)
            labels = np.array([
                1 if sum([d["labels_SQuAD"][i], d["labels_LLM"][i], d["labels_Gemini"][i]]) >= 2 else 0
                for i in range(len(d["labels_SQuAD"]))
            ])
        else:
            raise ValueError(f"Invalid label_type: {label_type}. Choose 'SQuAD', 'LLM', 'Gemini', or 'majority'.")

        # Select measure scores
        if measure_type == 'SE':
            y_score = np.array(d["semantic_entropy"])
        elif measure_type == 'p_true':
            y_score = np.array(d["p_true"])
        elif measure_type == 'PE':
            y_score = np.array(d["PE"])
        elif measure_type == 'Ln-PE':
            y_score = np.array(d["Ln-PE"])
        else:
            raise ValueError(f"Invalid measure_type: {measure_type}. Choose 'SE', 'p_true', 'PE', or 'Ln-PE'.")

        rejection_percentages = np.linspace(0.1, 1, 20)  
        rej_acc = [
            accuracy_at_quantile(labels, y_score, q)
            for q in rejection_percentages
        ]
        rej_acc_list.append(rej_acc)


        dx = rejection_percentages[1] - rejection_percentages[0]
        aurac = np.sum(np.array(rej_acc) * dx)  
        aurac_list.append(aurac)
        result = f"{d.info.description}-{measure_type:20} dataset: {aurac:8.4f}"
        print(result)
        results.append(result)
    
    # Save results to a .txt file
    with open(save_path, "a") as f:
        f.write('\n')
        f.write("\n".join(results))
