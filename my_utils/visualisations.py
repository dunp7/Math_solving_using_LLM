"""Visualise metrics"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def visualise_SE_distribution(models_names, datasets_names, results, colours, location= "results/figures"):
    """Creates a grouped box plot to visualize the distribution of semantic entropy (SE) values for various models 
       across datasets

    Parameters:
        models_names (list): A list of model names corresponding to the results
        datasets_names (list): A list of dataset names for which AUROC scores are computed
        results (list): A nested list containing AUROC scores for each model and dataset
        colours (dict): A dictionary mapping model names to their respective colors

    Output:
        Saves the plot as `results/figures/SE_distribution.png`

    Returns:
        None
    """

    reorder = []
    for d in range(4):
        for m in range(int(len(results)/4)):
            reorder.append(results[m*4+d])

    results = reorder

    groups = len(datasets_names)
    boxes_per_group = len(models_names)

    positions = [] # Set positions for the boxes (manually group them)
    for i in range(groups):
        positions += [i * (boxes_per_group + 1) + j for j in range(1, boxes_per_group + 1)]

    plt.figure(figsize=(16, 6))

    # Create the boxplot - plot one box at a time with the correct color
    for i in range(groups):  # Loop through each dataset group
        for j in range(boxes_per_group):  # Loop through boxes within the group
            idx = i * boxes_per_group + j  # Calculate overall index
            colour = colours[models_names[j]]  # Get the color for this model
            plt.boxplot(
                [results[idx]],  # Boxplot for this model's result
                positions=[positions[idx]],  # Position for this box
                patch_artist=True,
                boxprops=dict(facecolor=colour, color=colour),
                medianprops=dict(color="black"),
                widths=0.5
            )

    plt.xticks([i * (boxes_per_group + 1) + (boxes_per_group / 2) for i in range(groups)], datasets_names)
    num_columns = len(models_names) // 3 + (len(models_names) % 3)
    legend_patches = [mpatches.Patch(color=colours[model], label=model) for model in models_names]
    plt.legend(handles=legend_patches, title="Entailment Models", loc="lower center", 
               bbox_to_anchor=(0.5, -0.32), ncol=num_columns) # Place legend outside to the right
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplots_adjust(right=0.7)  # Shrink the plot to make space for the legend
    plt.title('Distribution of Semantic Entropy')
    plt.xlabel('Datatsets')
    plt.ylabel('Semantic Entropy')
    plt.savefig(location+'/SE_distribution.png', bbox_inches='tight')
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