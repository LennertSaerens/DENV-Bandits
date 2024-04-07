# Implementation of experiments with Thomson Sampling for Multi-Objective Multi-Armed Bandits
import numpy as np
import plotting
from bandits.ThompsonSampling import ParetoThompsonSamplingBandit, LinearScalarizedThompsonSamplingBandit
from bandits.UCB import ParetoUCB1Bandit, LinearScalarizedUCB1Bandit

num_runs = 10  # Number of experiments M
horizon = 5_000  # Number of time steps T

# Configuration for the first experiment
e1_arms = [(0.55, 0.5), (0.53, 0.51), (0.52, 0.54), (0.5, 0.57), (0.51, 0.51), (0.5, 0.5)] + 14 * [(0.48, 0.48)]
e1_num_arms = len(e1_arms)
e1_num_objectives = len(e1_arms[0])
e1_pareto_arms = [0, 1, 2, 3]
e1_weights = [(x, 1 - x) for x in np.linspace(0, 1, 11)]
e1_cheby_refs = np.random.uniform(0, 0.1, 2)


def calculate_pareto_regret(arm, arms, pareto_arms):
    """
    Calculate the pareto regret for reward that was received at time step t for pulling arm.
    The regret is zero if the arm is pareto optimal. Otherwise, the regret is the distance between the success
    rates of the arm and the success rates of the closest pareto optimal arm.
    :param pareto_arms: The pareto optimal arms.
    :param arms: The arms of the bandit.
    :param arm: The arm that was pulled.
    :return: The pareto regret.
    """
    if arm in pareto_arms:
        return 0
    else:
        pareto_distances = [np.sqrt(sum((np.array(arms[arm]) - np.array(arms[pareto_arm])) ** 2)) for pareto_arm in pareto_arms]
        return min(pareto_distances)


def calculate_unfairness_regret(arm_pulls, pareto_arms):
    """
    Calculate the unfairness regret that was received at time step t for pulling arm. The unfairness regret
    measure is the Shannon entropy which is a measure of the disorder of the frequency of selecting the optimal arms in
    the pareto front.
    :param pareto_arms: The pareto optimal arms.
    :param arm_pulls: The number of times each arm has been pulled.
    :return: The unfairness regret.
    """
    # Total number of pulls across all arms
    total_pulls = sum(arm_pulls)
    # Number of times each pareto optimal arm has been pulled
    pareto_arm_pulls = [arm_pulls[pareto_arm] for pareto_arm in pareto_arms]
    total_pareto_pulls = sum(pareto_arm_pulls)
    if total_pulls == 0 or total_pareto_pulls == 0:
        return 0
    else:
        frequencies = [pulls / total_pulls for pulls in pareto_arm_pulls]
        # Return 0 if any of the frequencies are 0
        if 0 in frequencies:
            return 0
        return -(1 / total_pareto_pulls) * sum(frequencies * np.log(frequencies))


def pull(arm, arms, num_objectives):
    """
    Pull an arm of the bandit and return the reward for each objective. The rewards ri of arm i are drawn from
    a Bernoulli distribution. The success rate for each objective is given by the arms list.
    :param num_objectives: The number of objectives.
    :param arms: The arms of the bandit.
    :param arm: The arm to pull.
    :return: The reward for each objective.
    """
    return [int(np.random.random() < arms[arm][o]) for o in range(num_objectives)]


def run_experiment(num_arms, num_objectives, arms, pareto_arms, weights, log=False):
    setup = {
        "Pareto Thompson Sampling": {
            "agent": ParetoThompsonSamplingBandit(num_arms, num_objectives),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)]
        },
        "Linear Scalarized Thompson Sampling": {
            "agent": LinearScalarizedThompsonSamplingBandit(num_arms, num_objectives, weights),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)]
        },
        "Pareto UCB1": {
            "agent": ParetoUCB1Bandit(num_arms, num_objectives, 1),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)]
        },
        "Linear Scalarized UCB1": {
            "agent": LinearScalarizedUCB1Bandit(num_arms, num_objectives, weights, 1),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)]
        },
    }

    for algorithm in setup:
        agent = setup[algorithm]["agent"]
        cumulative_pareto_regrets = setup[algorithm]["cumulative_pareto_regrets"]
        cumulative_unfairness_regrets = setup[algorithm]["cumulative_unfairness_regrets"]

        for experiment in range(num_runs):
            print(f"Experiment {experiment} for algorithm {algorithm}")
            arm_pulls = np.zeros(num_arms)  # Number of times each arm has been pulled
            agent.reset()

            for t in range(horizon):
                # Choose an arm
                arm = agent.choose_arm()
                # Pull the arm and receive the reward
                reward = pull(arm, arms, num_objectives)
                arm_pulls[arm] += 1
                # Learn from the reward
                agent.learn(arm, reward)

                # Calculate the pareto regret and the unfairness regret
                pareto_regret = calculate_pareto_regret(arm, arms, pareto_arms)
                unfairness_regret = calculate_unfairness_regret(arm_pulls, pareto_arms)

                if log:
                    print(
                        f"t: {t}, arm: {arm}, reward: {reward}, pareto regret: {pareto_regret}, unfairness regret: {unfairness_regret}")

                # Update the cumulative pareto regret and the cumulative unfairness regret
                cumulative_pareto_regrets[experiment].append(
                    cumulative_pareto_regrets[experiment][-1] + pareto_regret if t > 0 else pareto_regret)
                cumulative_unfairness_regrets[experiment].append(
                    cumulative_unfairness_regrets[experiment][-1] + unfairness_regret if t > 0 else unfairness_regret)

    # Plot the cumulative pareto regrets and the cumulative unfairness regrets
    plotting.plot_regrets(setup)


if __name__ == "__main__":
    run_experiment(e1_num_arms, e1_num_objectives, e1_arms, e1_pareto_arms, e1_weights, log=False)
