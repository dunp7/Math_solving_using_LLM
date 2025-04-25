"""
P(true) baseline to compare with semantic entropy

based on https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/uncertainty_measures/p_true.py

"""
import tqdm









def generate_ptrue(
    datasets, data_ptrue_path, llm_tokenizer, entail_model, calculate_p_true_function, few_shot_prompt=None
):
    """Calculates P_true for questions in multiple datasets.

    Parameters:
        datasets (list): A list of datasets, where each dataset contains questions and previously generated answers.
        data_ptrue_path (str): The directory path where the updated datasets with P_true will be saved.
        entail_model (AutoModelForSequenceClassification or AutoModelForCausalLM): The model used for entailment evaluation.
        calculate_p_true_function (function): Function that calculates the p_true metric.
        few_shot_prompt (str, optional): A pre-constructed few-shot prompt for calculating p_true. Defaults to None.

    Returns:
        None: The results are directly saved to disk.
    """
    for dataset in datasets:
        all_ptrue_scores = []

        print(f"\nCalculating P_true for {dataset_copy.info.description} dataset...")

        for i in tqdm(range(len(dataset_copy))):
            question = dataset[i]["question"]
            generated_answers = dataset[i]["generated_answers_acc"]
            most_probable_answer = llm_tokenizer.decode(generated_answers["sequences"][0], skip_special_tokens=True)

            # Calculate P_true
            p_true_score = calculate_p_true_function(
                entail_model, question, most_probable_answer, 
                [ans["text"] for ans in generated_answers["sequences"]], few_shot_prompt
            )
            all_ptrue_scores.append(p_true_score)

        # Save results to dataset
        dataset_copy = dataset_copy.add_column("p_true_scores", all_ptrue_scores)
        dataset_copy.save_to_disk(data_ptrue_path + dataset_copy.info.description)
