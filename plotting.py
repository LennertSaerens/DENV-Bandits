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
        axs[0].fill_between(range(len(avg_cumulative_pareto_regrets)),
                            avg_cumulative_pareto_regrets - std_cumulative_pareto_regrets,
                            avg_cumulative_pareto_regrets + std_cumulative_pareto_regrets, alpha=0.2)
        axs[1].plot(avg_cumulative_unfairness_regrets, label=f"{algorithm}")
        axs[1].fill_between(range(len(avg_cumulative_unfairness_regrets)),
                            avg_cumulative_unfairness_regrets - std_cumulative_unfairness_regrets,
                            avg_cumulative_unfairness_regrets + std_cumulative_unfairness_regrets, alpha=0.2)
    axs[0].set_title("Cumulative Pareto Regrets")
    axs[0].set_xlabel("Time steps")
    axs[0].set_ylabel("Cumulative Pareto Regret")
    axs[0].legend()
    axs[1].set_title("Cumulative Unfairness Regrets")
    axs[1].set_xlabel("Time steps")
    axs[1].set_ylabel("Cumulative Unfairness Regret")
    axs[1].legend()
    plt.show()
