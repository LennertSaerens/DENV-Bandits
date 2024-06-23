import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from sklearn.preprocessing import MinMaxScaler

from bandits.Annealing import APBandit
from bandits.KnowledgeGradient import PKGBandit
from bandits.TTPFTS import NormalTTPFTSBandit
from bandits.ThompsonSampling import NormalPTSBandit
from bandits.UCB import PUCB1Bandit

data = pd.read_csv('results/Experiment1Extended.csv')

medical_burden = data['Medical Burden'].values.reshape(-1, 1)
monetary_cost = data['Monetary Cost'].values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_medical_burden = scaler.fit_transform(medical_burden)
normalized_monetary_cost = scaler.fit_transform(monetary_cost)

inverted_arms = [(normalized_medical_burden[i][0], normalized_monetary_cost[i][0]) for i in
                 range(len(normalized_medical_burden))]
print(inverted_arms)

# Invert the data for maximization
maximized_medical_burden = 1 - normalized_medical_burden
maximized_monetary_cost = 1 - normalized_monetary_cost

arms = [(maximized_medical_burden[i][0], maximized_monetary_cost[i][0]) for i in range(len(maximized_medical_burden))]
print(arms)

std = 0.1

reference_point = np.array([1, 1])

optimal_arms = [0, 5, 6, 8, 14, 30, 31, 32]

num_runs = 100
horizon = 30_000


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
    F = np.array([inverted_arms[arm] for arm in recommended])
    ind = HV(ref_point=reference_point)
    hv = ind.do(F)
    return hv


def pull_arm(arm):
    return [np.random.normal(arm[0], std), np.random.normal(arm[1], std)]


def run_experiment(num_arms, num_objectives, arms, pareto_indices, results_file=None, write=False):
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
        "TT PF Thompson Sampling": {
            "agent": NormalTTPFTSBandit(num_arms, num_objectives),
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

                bernoulli_metric = is_completely_cor_rec(recommended, pareto_indices)
                jaccard_metric = calculate_jaccard_similarity(recommended, pareto_indices)
                hypervolume = calc_hypervolume(recommended)

                if results_file is not None and write:
                    with open(results_file, "a") as file:
                        file.write(
                            f"{algorithm},{experiment},{t},{bernoulli_metric},{jaccard_metric},{hypervolume},{arm}\n")


if __name__ == '__main__':
    run_experiment(len(arms), 2, arms, optimal_arms, results_file='results/finalstd1.csv', write=True)
