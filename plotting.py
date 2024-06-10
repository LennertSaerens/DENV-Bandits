import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse

# Increase the font size of the plots
plt.rcParams.update({'font.size': 12})
# Change the font to a fancy serif font for use in a latex document
plt.rcParams.update({'font.family': 'serif'})


def plot_regrets(setup_dict):
    """
    Plot the evolution of the cumulative pareto regrets and the cumulative unfairness regrets for each of the different
    algorithms in the experimental setup in a single figure with two sub figures that sit side by side.
    The x-axis represents the time steps and the y-axis represents the cumulative regrets. The regrets are averaged over the experiments.
    :param setup_dict: The experimental setup dictionary.
    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for algorithm in setup_dict:
        cumulative_pareto_regrets = setup_dict[algorithm]["cumulative_pareto_regrets"]
        avg_cumulative_pareto_regrets = np.mean(cumulative_pareto_regrets, axis=0)
        std_cumulative_pareto_regrets = np.std(cumulative_pareto_regrets, axis=0)
        cumulative_unfairness_regrets = setup_dict[algorithm]["cumulative_unfairness_regrets"]
        avg_cumulative_unfairness_regrets = np.mean(cumulative_unfairness_regrets, axis=0)
        std_cumulative_unfairness_regrets = np.std(cumulative_unfairness_regrets, axis=0)
        axs[0].plot(avg_cumulative_pareto_regrets, label=f"{algorithm}")
        # Plot the 95% confidence interval for the cumulative pareto regrets
        axs[0].fill_between(range(len(avg_cumulative_pareto_regrets)),
                            avg_cumulative_pareto_regrets - 1.96 * std_cumulative_pareto_regrets / np.sqrt(
                                len(cumulative_pareto_regrets)),
                            avg_cumulative_pareto_regrets + 1.96 * std_cumulative_pareto_regrets / np.sqrt(
                                len(cumulative_pareto_regrets)),
                            alpha=0.2)
        axs[1].plot(avg_cumulative_unfairness_regrets, label=f"{algorithm}")
        # Plot the 95% confidence interval for the cumulative unfairness regrets
        axs[1].fill_between(range(len(avg_cumulative_unfairness_regrets)),
                            avg_cumulative_unfairness_regrets - 1.96 * std_cumulative_unfairness_regrets / np.sqrt(
                                len(cumulative_unfairness_regrets)),
                            avg_cumulative_unfairness_regrets + 1.96 * std_cumulative_unfairness_regrets / np.sqrt(
                                len(cumulative_unfairness_regrets)),
                            alpha=0.2)
    axs[0].set_title("Cumulative Pareto Regrets")
    axs[0].set_xlabel("Time steps")
    axs[0].set_ylabel("Cumulative Pareto Regret")
    axs[0].legend()
    axs[1].set_title("Cumulative Unfairness Regrets")
    axs[1].set_xlabel("Time steps")
    axs[1].set_ylabel("Cumulative Unfairness Regret")
    axs[1].legend()
    plt.show()


def plot_arms_pareto_front(arms, pareto_indices, plot_stds=False):
    """
    Plot the arms in the 2D objective space and highlight the Pareto front in the plot by plotting the Pareto optimal arms in a different color.
    If plot_stds is set to True, the standard deviations of the arms are also plotted as shaded ellipse around the mean.
    :param arms: The means of the arms for each objective.
    :param pareto_indices: The indices of the Pareto optimal arms.
    :param plot_stds: Whether to plot the standard deviations of the arms as well.
    :return: None
    """
    plt.scatter(arms[:, 0], arms[:, 2], color='red')
    # Annotate the arms with their index at an offset
    for i in range(len(arms)):
        plt.annotate(i, (arms[i, 0], arms[i, 2]), textcoords="offset points", xytext=(0, 5), ha='center')
    for pareto_index in pareto_indices:
        plt.scatter(arms[pareto_index, 0], arms[pareto_index, 2], color='green')
    if plot_stds:
        for arm in arms:
            ellipse = Ellipse((arm[0], arm[2]), width=arm[1], height=arm[3], alpha=0.05)
            plt.gca().add_patch(ellipse)
    plt.xlabel("Hospitalizations")
    plt.ylabel("Costs")
    plt.title("Arms in the 2D objective space")
    plt.show()


def plot_arm_pulls(setup, optimal_arms, total_pulls, show_arm_idxs=False):
    """
    Plot the frequency of pulling each arm for each algorithm in the experimental setup. Each algorithm has its own subplot within the big plot with 4 rows and 2 columns.
    Inside each subplot, the number of times the algorithm pulled each arm is plotted as a bar for each arm. Pareto optimal arms are highlighted in a different color.
    All other arms have the same color.
    :param total_pulls: The total number of times an arm was pulled for each algorithm. Used for frequency calculation.
    :param optimal_arms: The indices of the Pareto optimal arms.
    :param setup: The experimental setup dictionary.
    :param show_arm_idxs: Whether to show the arm indices on the x-axis.
    :return: None
    """
    fig, axs = plt.subplots(4, 2, figsize=(20, 20))
    for i, algorithm in enumerate(setup):
        ax = axs[i // 2, i % 2]
        arm_pulls = setup[algorithm]["arm_pulls"]
        avg_arm_pulls = np.mean(arm_pulls, axis=0) / total_pulls
        std_arm_pulls = np.std(arm_pulls, axis=0) / total_pulls
        ax.bar(range(len(avg_arm_pulls)), avg_arm_pulls, yerr=1.96 * std_arm_pulls / np.sqrt(len(arm_pulls)))
        ax.set_title(f"{algorithm}")
        if show_arm_idxs:
            ax.set_xticks(range(len(avg_arm_pulls)))
        # Highlight the Pareto optimal arms in the plot
        for optimal_arm in optimal_arms:
            ax.get_children()[optimal_arm].set_color('green')
    # Show 'Frequency' on the y-axis of all plots in the first column
    for i in range(4):
        axs[i, 0].set_ylabel("Frequency")
    # Show 'Arm index' on the x-axis of all plots in the bottom row
    axs[3, 0].set_xlabel("Arm index")
    axs[3, 1].set_xlabel("Arm index")
    plt.show()


def plot_arm_pulls_2(setup, optimal_arms, total_pulls):
    """
    Plot the frequency of pulling each arm for each algorithm in the experimental setup. Each algorithm has its own subplot within the big plot with 2 rows and 4 columns.
    Inside each subplot, the number of times the algorithm pulled each arm is plotted as a bar for each arm. Pareto optimal arms are highlighted in a different color.
    All other arms have the same color.
    :param total_pulls: The total number of times an arm was pulled for each algorithm. Used for frequency calculation.
    :param optimal_arms: The indices of the Pareto optimal arms.
    :param setup: The experimental setup dictionary.
    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for i, algorithm in enumerate(setup):
        ax = axs[i]
        arm_pulls = setup[algorithm]["arm_pulls"]
        avg_arm_pulls = np.mean(arm_pulls, axis=0) / total_pulls
        std_arm_pulls = np.std(arm_pulls, axis=0) / total_pulls
        ax.bar(range(len(avg_arm_pulls)), avg_arm_pulls, yerr=1.96 * std_arm_pulls / np.sqrt(len(arm_pulls)))
        ax.set_title(f"{algorithm}")
        ax.set_xticks(range(len(avg_arm_pulls)))
        # Highlight the Pareto optimal arms in the plot
        for optimal_arm in optimal_arms:
            ax.get_children()[optimal_arm].set_color('green')
    # Show 'Frequency' on the y-axis of all plots
    for i in range(2):
        axs[i].set_ylabel("Frequency")
    # Show 'Arm index' on the x-axis of all plots
    for i in range(2):
        axs[i].set_xlabel("Arm index")
    plt.show()


def plot_arm_pulls_4(setup, optimal_arms, total_pulls):
    """
    Plot the frequency of pulling each arm for each algorithm in the experimental setup. Each algorithm has its own subplot within the big plot with 2 rows and 4 columns.
    Inside each subplot, the number of times the algorithm pulled each arm is plotted as a bar for each arm. Pareto optimal arms are highlighted in a different color.
    All other arms have the same color.
    :param total_pulls: The total number of times an arm was pulled for each algorithm. Used for frequency calculation.
    :param optimal_arms: The indices of the Pareto optimal arms.
    :param setup: The experimental setup dictionary.
    :return: None
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    for i, algorithm in enumerate(setup):
        ax = axs[i // 2, i % 2]
        arm_pulls = setup[algorithm]["arm_pulls"]
        avg_arm_pulls = np.mean(arm_pulls, axis=0) / total_pulls
        std_arm_pulls = np.std(arm_pulls, axis=0) / total_pulls
        ax.bar(range(len(avg_arm_pulls)), avg_arm_pulls, yerr=1.96 * std_arm_pulls / np.sqrt(len(arm_pulls)))
        ax.set_title(f"{algorithm}")
        ax.set_xticks(range(len(avg_arm_pulls)))
        # Highlight the Pareto optimal arms in the plot
        for optimal_arm in optimal_arms:
            ax.get_children()[optimal_arm].set_color('green')
    # Show 'Frequency' on the y-axis of all plots in the first column
    for i in range(2):
        axs[i, 0].set_ylabel("Frequency")
    # Show 'Arm index' on the x-axis of all plots in the second row
    for i in range(2):
        axs[1, i].set_xlabel("Arm index")
    plt.show()


def plot_arm_pulls_single(setup, algorithm_name, optimal_arms, total_pulls):
    """
    Plot the frequency of pulling each arm for a single algorithm in the experimental setup.
    The number of times the algorithm pulled each arm is plotted as a bar for each arm. Pareto optimal arms are highlighted in a different color.
    All other arms have the same color.
    :param setup: The experimental setup dictionary.
    :param algorithm_name: The name of the algorithm to plot.
    :param optimal_arms: The indices of the Pareto optimal arms.
    :param total_pulls: The total number of times an arm was pulled for the algorithm. Used for frequency calculation.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    arm_pulls = setup[algorithm_name]["arm_pulls"]
    avg_arm_pulls = np.mean(arm_pulls, axis=0) / total_pulls
    std_arm_pulls = np.std(arm_pulls, axis=0) / total_pulls
    ax.bar(range(len(avg_arm_pulls)), avg_arm_pulls, yerr=1.96 * std_arm_pulls / np.sqrt(len(arm_pulls)))
    ax.set_title(f"{algorithm_name}")
    ax.set_xticks(range(len(avg_arm_pulls)))
    # Highlight the Pareto optimal arms in the plot
    for optimal_arm in optimal_arms:
        ax.get_children()[optimal_arm].set_color('green')
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Arm index")
    plt.show()


def plot_bernoulli_metric(setup):
    """
    Plot the evolution of the Bernoulli metric for a single algorithm in the experimental setup. The Bernoulli metric is the fraction of times the algorithm pulls a Pareto optimal arm.
    The x-axis represents the time steps and the y-axis represents the Bernoulli metric. The metric is averaged over the experiments.
    :param setup: The experimental setup dictionary.
    :param algorithm_name: The name of the algorithm to plot.
    :return: None
    """
    for algorithm_name in setup:
        bernoulli_metrics = setup[algorithm_name]["cor_rec_ber"]
        avg_bernoulli_metrics = np.mean(bernoulli_metrics, axis=0)
        plt.plot(avg_bernoulli_metrics, label=f"{algorithm_name}")
        plt.title(f"Bernoulli metric for {algorithm_name}")
        plt.xlabel("Time steps")
        plt.ylabel("Bernoulli metric")
        plt.legend()
        plt.show()


def plot_jaccard_metric(setup):
    """
    Plot the evolution of the Jaccard metric for a single algorithm in the experimental setup. The Jaccard metric is the Jaccard similarity between the set of Pareto optimal arms and the set of arms recommended by the algorithm.
    The x-axis represents the time steps and the y-axis represents the Jaccard metric. The metric is averaged over the experiments.
    :param setup: The experimental setup dictionary.
    :return: None
    """
    for algorithm_name in setup:
        jaccard_metrics = setup[algorithm_name]["jaccard_sim"]
        avg_jaccard_metrics = np.mean(jaccard_metrics, axis=0)
        plt.plot(avg_jaccard_metrics, label=f"{algorithm_name}")
        plt.title(f"Jaccard distance metric for {algorithm_name}")
        plt.xlabel("Time steps")
        plt.ylabel("Jaccard metric")
        plt.legend()
        plt.show()
