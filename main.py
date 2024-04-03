# Implementation of experiments with Thomson Sampling for Multi-Objective Multi-Armed Bandits
import numpy as np

import MOTSBandits
import helpers
import plotting

num_experiments = 1000  # Number of experiments M
horizon = 1000  # Number of time steps T
num_objectives = 5  # Number of objectives K
num_optimal = 7  # Number of optimal arms
num_suboptimal = 13  # Number of suboptimal arms
num_arms = num_optimal + num_suboptimal  # Number of arms N

W = helpers.create_random_weights(num_objectives, 10)

# Success rates for each of the objectives for each of the arms
arms = helpers.create_arms(num_optimal, num_suboptimal, num_objectives)
print(f"The arms are: {arms}")

# Calculate the true Pareto optimal arms based on the success rates
pareto_arms = [arm for arm in range(num_arms) if not any(np.all(np.array(arms[arm]) < np.array(arms[other_arm])) for other_arm in range(num_arms) if arm != other_arm)]
print(f"The true Pareto optimal arms are: {pareto_arms}, number of Pareto optimal arms: {len(pareto_arms)}")


def calculate_pareto_regret(arm):
    """
    Calculate the pareto regret for reward that was received at time step t for pulling arm.
    The regret is zero if the arm is pareto optimal. Otherwise, the regret is the distance between the success
    rates of the arm and the success rates of the closest pareto optimal arm.
    :param arm: The arm that was pulled.
    :return: The pareto regret.
    """
    if arm in pareto_arms:
        return 0
    else:
        pareto_distances = [np.sqrt(sum((np.array(arms[arm]) - np.array(arms[pareto_arm])) ** 2)) for pareto_arm in
                            pareto_arms]
        return min(pareto_distances)


def calculate_unfairness_regret(arm_pulls):
    """
    Calculate the unfairness regret that was received at time step t for pulling arm. The unfairness regret
    measure is the Shannon entropy which is a measure of the disorder of the frequency of selecting the optimal arms in
    the pareto front.
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


def pull(arm):
    """
    Pull an arm of the bandit and return the reward for each objective. The rewards ri of arm i are drawn from
    a Bernoulli distribution. The success rate for each objective is given by the arms list.
    :param arm: The arm to pull.
    :return: The reward for each objective.
    """
    return [int(np.random.random() < arms[arm][o]) for o in range(num_objectives)]


def main(log=False):
    setup = {
        "Pareto Thompson Sampling": {
            "agent": MOTSBandits.ParetoThompsonSamplingBandit(num_arms, num_objectives),
            "cumulative_pareto_regrets": [[] for _ in range(num_experiments)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_experiments)]
        },
        "Linear Scalarized Thompson Sampling": {
            "agent": MOTSBandits.LinearScalarizedThompsonSamplingBandit(num_arms, num_objectives, W),
            "cumulative_pareto_regrets": [[] for _ in range(num_experiments)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_experiments)]
        }
    }

    for algorithm in setup:
        agent = setup[algorithm]["agent"]
        cumulative_pareto_regrets = setup[algorithm]["cumulative_pareto_regrets"]
        cumulative_unfairness_regrets = setup[algorithm]["cumulative_unfairness_regrets"]

        for experiment in range(num_experiments):
            print(f"Experiment {experiment} for algorithm {algorithm}")
            arm_pulls = np.zeros(num_arms)  # Number of times each arm has been pulled
            agent.reset()

            for t in range(horizon):
                # Choose an arm
                arm = agent.choose_arm()
                # Pull the arm and receive the reward
                reward = pull(arm)
                arm_pulls[arm] += 1
                # Learn from the reward
                agent.learn(arm, reward)

                # Calculate the pareto regret and the unfairness regret
                pareto_regret = calculate_pareto_regret(arm)
                unfairness_regret = calculate_unfairness_regret(arm_pulls)

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
    main()
