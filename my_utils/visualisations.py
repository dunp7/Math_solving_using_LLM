"""Visualise metrics"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def visualise_aur_results(models_names, datasets_names, results, colours, metric_name, location = "results/figures"):
    """Creates a bar plot to visualize F1/AUROC/AURAC scores across different models and datasets

    Parameters:
        models_names (list): A list of model names corresponding to the results
        datasets_names (list): A list of dataset names for which AUROC/AURAC scores are computed
        results (list): A nested list containing AUROC/AURAC scores for each model and dataset
        colours (dict): A dictionary mapping model names to their respective colors
        metric_name(string): Name of metric (`F1`,`AUROC` or `AURAC`) 

    Output:
        Saves the plot as `results/figures/{metric_name}_scores.png`

    Returns:
        None
    """


    # Check if the directory exists
    if not os.path.exists(location):
        os.makedirs(location)  # Create the directory if it doesn't exist

    spacing = 2.4  # Adjust this value to control the space between datasets
    x = np.arange(0, len(datasets_names) * spacing, spacing)
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (model, result) in enumerate(zip(models_names, results)):
        colour = colours.get(model, 'grey')
        ax.bar(x + i * width, result, width, label=model, color=colour)

    ax.set_xlabel("Datasets")
    ax.set_ylabel(metric_name.upper() + " Score")
    ax.set_title(metric_name.upper() + " Scores for Sentence-length Experiment")
    ax.set_xticks(x + width * (len(models_names) - 1) / 2)
    ax.set_xticklabels(datasets_names)
    ax.grid(True, linestyle=":", linewidth=0.5)
    num_columns = len(models_names) // 3 + (len(models_names) % 3)
    ax.legend(title="Entailment Models", loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=num_columns)
    ax.set_ylim(0, 0.8)

    plt.tight_layout()
    plt.savefig(location + "/" + metric_name.lower() + "_scores.png")
    plt.show()


def visualise_aur_percentages_results(models_names, datasets_names, results, colours,location= "results/figures"):
    """ Creates a bar plot to visualize rejection accuracy across different rejection percentages, for various models and datasets
    
    Parameters:
        models_names (list): A list of model names corresponding to the results
        datasets_names (list): A list of dataset names for which rejection accuracies are computed
        results (list): A nested list (or array) containing rejection accuracies for each model, dataset, and rejection percentage
                        Shape: [num_models, num_datasets, num_percentages]
        colours (dict): A dictionary mapping model names to their respective colors

    Output:
        Saves the plot as `results/figures/aurac_percentages.png`

    Returns:
        None
    """

    # Check if the directory exists
    if not os.path.exists(location):
        os.makedirs(location)  # Create the directory if it doesn't exist
    fig, axes = plt.subplots(len(datasets_names), 1, figsize=(12, 12), sharex=True, sharey=True)
    percentages = np.arange(10, 100, 10)
    results = np.array(results)

    # Ensure axes is always a list
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, dataset in enumerate(datasets_names):
        ax = axes[i]  # Get the axis for the current dataset
        x = np.arange(len(percentages))  # x positions for the bars
        width = 0.09  # Width of each bar

        for j, model in enumerate(models_names):
            # Extract data for the current model and dataset
            y_values = results[j, i, :][::-1]  # Shape: [percentages]
            ax.bar(x + j * width, y_values, width, label=model if i == 0 else "", color=colours[model])

        # Format subplot
        ax.set_title(dataset, fontsize=12, pad=10)
        ax.set_ylabel("Rejection Accuracy", fontsize=10)
        ax.set_xticks(x + width * (len(models_names) / 2 - 0.5))  # Center x-ticks
        ax.set_xticklabels([f"{p}%" for p in percentages])
        ax.set_ylim(0, 0.8)
        ax.grid(axis='y')

    # Single xlabel for the last axis
    axes[-1].set_xlabel("Rejection Percentages")

    # Add shared legend
    num_columns = len(models_names) // 2 + (len(models_names) % 2)
    fig.legend(models_names, title="Entailment Models", loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=num_columns)
    fig.suptitle("Rejection Accuracies for Different Rejection Percentages", fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(location + "/aurac_percentages.png")
    plt.show()


def visualise_SE_distribution(models_names, dataset_name, results, colours, location = "results/figures"):
    """
    Creates a grouped box plot to visualize the distribution of semantic entropy (SE) values 
    for various models on a single dataset.

    Parameters:
        models_names (list): A list of model names corresponding to the results
        dataset_name (str): The name of the dataset for which SE values are visualized
        results (list): A list containing SE values for each model
        colours (dict): A dictionary mapping model names to their respective colors
        location (str): Directory to save the output figure

    Output:
        Saves the plot as `SE_distribution.png` in the specified location.

    Returns:
        None
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Ensure the directory exists
    if not os.path.exists(location):
        os.makedirs(location)

    # Calculate positions for the box plots
    boxes_per_group = len(models_names)
    positions = [i for i in range(boxes_per_group)]

    plt.figure(figsize=(8, 6))

    # Create the box plots for each model
    for idx, model in enumerate(models_names):
        colour = colours[model]
        plt.boxplot(
            [results[idx]],  # Boxplot for this model's result
            positions=[positions[idx]],  # Position for this box
            patch_artist=True,
            boxprops=dict(facecolor=colour, color=colour),
            medianprops=dict(color="black"),
            widths=0.5
        )

    # Customize the plot
    plt.xticks(positions, models_names, rotation=45)
    legend_patches = [mpatches.Patch(color=colours[model], label=model) for model in models_names]
    plt.legend(handles=legend_patches, title="Entailment Models", loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.title("Distribution of Semantic Entropy")
    plt.xlabel("Models")
    plt.ylabel("Semantic Entropy")

    # Save and show the plot
    plt.savefig(os.path.join(location, "SE_distribution.png"), bbox_inches="tight")
    plt.show()



def visualise_SE_mean_std(models_names, datasets_names, results, colours, location= "results/figures"):
    """Visualizes the mean and standard deviation of semantic entropy (SE) values for different models and datasets 
       using line plots with std bands

    Parameters:
        models_names (list): A list of model names corresponding to the results
        datasets_names (list): A list of dataset names for which AUROC scores are computed
        results (list): A nested list containing AUROC scores for each model and dataset
        colours (dict): A dictionary mapping model names to their respective colors

    Output:
        Saves the plot as `results/figures/SE_mean_std.png`

    Returns:
        None
    """

    # Check if the directory exists
    if not os.path.exists(location):
        os.makedirs(location)  # Create the directory if it doesn't exist

    grouped_results = {dataset: {} for dataset in datasets_names}

    index = 0
    for model in models_names:
        for dataset in datasets_names:
            grouped_results[dataset][model] = results[index]
            index += 1

    means = []
    stds = []
    for dataset in datasets_names:
        dataset_means = []
        dataset_stds = []
        for model in models_names:
            model_results = grouped_results[dataset][model]
            dataset_means.append(np.mean(model_results))
            dataset_stds.append(np.std(model_results))
        means.append(dataset_means)
        stds.append(dataset_stds)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets_names))  # Dataset positions

    for i, model in enumerate(models_names):
        
        mean_values = [means[j][i] for j in range(len(datasets_names))]
        std_values = [stds[j][i] for j in range(len(datasets_names))]
        ax.plot(x, mean_values, label=model, marker='o', linestyle='-', color=colours[model])
        # ax.plot(x, [m - s for m, s in zip(mean_values, std_values)], label=model, marker='*', linestyle='--', color=colours[model])
        # ax.plot(x, [m + s for m, s in zip(mean_values, std_values)], label=model, marker='+', linestyle='--', color=colours[model])
        ax.fill_between(x, 
                        [m - s for m, s in zip(mean_values, std_values)], 
                        [m + s for m, s in zip(mean_values, std_values)], 
                        color=colours[model], alpha=0.2)

    ax.set_xlabel("Datasets")
    ax.set_ylabel("Semantic Entropy (Mean ± Std)")
    ax.set_title("Mean and Standard Deviation of Semantic Entropy")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_names)
    ax.grid(True, linestyle=":", linewidth=0.5)
    num_columns = len(models_names) // 3 + (len(models_names) % 3)
    ax.legend(title="Entailment Models", loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=num_columns)
    ax.set_ylim(-0.3, 1.3)

    plt.tight_layout()
    plt.savefig(location + "/SE_mean_std.png")
    plt.show()



def create_description_txt(gen_model, acc_model, prompt, sample, dataset, location= "results/"):
    "Create a description.txt for the result"
        # Ensure the directory exists
    if not os.path.exists(location):
        os.makedirs(location)

    with open(f"{location}description.txt", "w") as f:
        f.write(f"Generated model: {gen_model}\n")
        f.write(f"Accuracy model: {acc_model}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Sample: {sample}\n")
        f.write(f"Dataset: {dataset}\n")

    print(f"Description file created: {location}description.txt")    
