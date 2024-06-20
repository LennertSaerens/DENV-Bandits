import numpy as np
from pymoo.indicators.hv import HV

from bandits.Annealing import APBandit
from bandits.KnowledgeGradient import PKGBandit
from bandits.ThompsonSampling import NormalPTSBandit
from bandits.UCB import PUCB1Bandit

from plotting import plot_arms_PFI_setting

pareto_optimal_arms = [(1, 5), (1.5, 3.5), (3, 3), (3.5, 1.5), (5, 1)]
suboptimal_arms = [(0.5, 4), (1, 3), (1, 2), (1, 1), (2, 1), (2.5, 2.5), (3, 1), (4, 0.5)]
arms = pareto_optimal_arms + suboptimal_arms
pareto_indices = [arms.index(arm) for arm in pareto_optimal_arms]
std = 1

# transform each arm by inverting all the means
inverted_arms = [(5 - arm[0], 5 - arm[1]) for arm in arms]

num_runs = 100
horizon = 5_000


def pull_arm(arm):
    return [np.random.normal(arm[0], std), np.random.normal(arm[1], std)]


def is_completely_cor_rec(recommended, pareto_arms):
    """
    Check if the recommended arms are completely correct and recommended.
    :param recommended: The recommended arms.
    :param pareto_arms: The pareto optimal arms.
    :return: True if the recommended arms are completely correct and recommended, False otherwise.
    """
    return int(set(recommended) == set(pareto_arms))


def calculate_jaccard_similarity(recommended, pareto_arms):
    """
    Calculate the Jaccard similarity between the recommended arms and the pareto optimal arms.
    :param recommended: The recommended arms.
    :param pareto_arms: The pareto optimal arms.
    :return: The Jaccard similarity.
    """
    return len(set(recommended).intersection(set(pareto_arms))) / len(set(recommended).union(set(pareto_arms)))


def calc_hypervolume(recommended):
    """
    Calculate the hypervolume of the recommended arms using pymoo.
    :param recommended: The recommended arms.
    :return: The hypervolume.
    """
    ref_point = np.array([6, 6])
    F = np.array([inverted_arms[arm] for arm in recommended])
    ind = HV(ref_point=ref_point)
    hv = ind.do(F)
    return hv


def run_PFI_experiment(num_arms, num_objectives, arms, pareto_arms, results_file, write=False):
    setup = {
        "Pareto UCB1": {
            "agent": PUCB1Bandit(num_arms, num_objectives, 0.3),
        },
        "Pareto Thompson Sampling": {
            "agent": NormalPTSBandit(num_arms, num_objectives),
        },
        "Pareto Knowledge Gradient": {
            "agent": PKGBandit(num_arms, num_objectives, horizon, 3),
        },
        "Annealing Pareto": {
            "agent": APBandit(num_arms, num_objectives, horizon, 3, 1, 0.999),
        },
    }

    for algorithm in setup:
        agent = setup[algorithm]["agent"]

        for experiment in range(num_runs):
            print(f"Experiment {experiment} for algorithm {algorithm}")
            agent.reset()

            for t in range(horizon):
                arm = agent.choose_arm()
                reward = pull_arm(arms[arm])
                agent.learn(arm, reward)
                recommended = agent.get_top_arms()

                bernoulli_metric = is_completely_cor_rec(recommended, pareto_arms)
                jaccard_metric = calculate_jaccard_similarity(recommended, pareto_arms)
                hypervolume = calc_hypervolume(recommended)

                if results_file is not None and write:
                    with open(results_file, "a") as file:
                        file.write(f"{algorithm},{experiment},{t},{bernoulli_metric},{jaccard_metric},{hypervolume}\n")


if __name__ == '__main__':
    # plot_arms_PFI_setting(inverted_arms, pareto_indices, std)
    run_PFI_experiment(len(arms), 2, arms, pareto_indices, "results/PFI_results.csv", write=True)
