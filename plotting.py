import matplotlib.pyplot as plt
import numpy as np


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
                            avg_cumulative_pareto_regrets - 1.96 * std_cumulative_pareto_regrets / np.sqrt(len(cumulative_pareto_regrets)),
                            avg_cumulative_pareto_regrets + 1.96 * std_cumulative_pareto_regrets / np.sqrt(len(cumulative_pareto_regrets)),
                            alpha=0.2)
        axs[1].plot(avg_cumulative_unfairness_regrets, label=f"{algorithm}")
        # Plot the 95% confidence interval for the cumulative unfairness regrets
        axs[1].fill_between(range(len(avg_cumulative_unfairness_regrets)),
                            avg_cumulative_unfairness_regrets - 1.96 * std_cumulative_unfairness_regrets / np.sqrt(len(cumulative_unfairness_regrets)),
                            avg_cumulative_unfairness_regrets + 1.96 * std_cumulative_unfairness_regrets / np.sqrt(len(cumulative_unfairness_regrets)),
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


def plot_arms_pareto_front(arms, pareto_indices):
    """
    Plot the arms in the 2D objective space and highlight the Pareto front in the plot by plotting the Pareto optimal arms in a different color.
    :param arms: The means of the arms for each objective.
    :param pareto_indices: The indices of the Pareto optimal arms.
    :return: None
    """
    plt.scatter(arms[:, 0], arms[:, 1])
    # Annotate the arms with their index at an offset
    for i in range(len(arms)):
        plt.annotate(i, (arms[i, 0], arms[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center')
    for pareto_index in pareto_indices:
        plt.scatter(arms[pareto_index, 0], arms[pareto_index, 1], color='green')
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Arms in the 2D objective space")
    plt.show()


def plot_arm_pulls(setup):
    """
    Plot the frequency of pulling each arm for each algorithm in the experimental setup.
    :param setup: The experimental setup dictionary.
    :return: None
    """
    for algorithm in setup:
        arm_pulls = setup[algorithm]["arm_pulls"]
        avg_arm_pulls = np.mean(arm_pulls, axis=0)
        plt.plot(avg_arm_pulls, label=f"{algorithm}")
    plt.xlabel("Arm")
    plt.ylabel("Frequency")
    plt.title("Frequency of pulling each arm")
    plt.legend()
    plt.show()
