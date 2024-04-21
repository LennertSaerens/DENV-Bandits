import numpy as np
import pandas as pd

from bandits.Annealing import APBandit
from bandits.KnowledgeGradient import PKGBandit, LSKGArmsBandit, LSKGObjectivesBandit
from bandits.ThompsonSampling import NormalPTSBandit, NormalLSTSBandit
from bandits.UCB import PUCB1Bandit, LSUCB1Bandit
from main import calculate_unfairness_regret
from plotting import plot_regrets, plot_arms_pareto_front, plot_arm_pulls

# EXPERIMENTAL SETUP PARAMETERS
num_runs = 100  # Number of experiments M
horizon = 30_000  # Number of time steps T

# LOADING DATA
# use the contents of the Experiment0.csv file located in the results directory as a pandas dataframe
df = pd.read_csv('results/Experiment0.csv')

# divide the 'Costs' column by 10000
df['Costs'] = df['Costs'] / 10000

# Set zero cost for no vaccination to 1 to avoid division by zero
df.loc[0, 'Costs'] = 1

# Invert the values of the 'Hospitalizations' column to go from minimization to maximization
df['Hospitalizations'] = 1 / df['Hospitalizations']
df['Hosp error'] = 30
df['Costs'] = 1 / df['Costs']
df['Cost error'] = 0.3

# Each arm is a tuple of the form (hospitalizations, hospitalizations error, costs, costs error)
# = (mean objective 1, std objective 1, mean objective 2, std objective 2)
arms = [(df['Hospitalizations'][i], df['Hosp error'][i], df['Costs'][i], df['Cost error'][i]) for i in range(len(df))]

num_arms = len(arms)
num_objectives = len(arms[0]) // 2
pareto_arms = [0, 1, 4, 10, 16]
weights = [(x, 1 - x) for x in np.linspace(0, 1, 11)]

# Plot the arms
plot_arms_pareto_front(np.array([[arm[0], arm[2]] for arm in arms]), pareto_arms)


def pull(arm, arms):
    """
    Pull an arm of the bandit and return the reward for each objective. The rewards ri of arm i are drawn from
    a normal distribution with mean mi and standard deviation si.
    :param arm: The arm to pull.
    :param arms: The arms of the bandit.
    :return: The reward for each objective.
    """
    obj1 = np.random.normal(arms[arm][0], arms[arm][1])
    obj2 = np.random.normal(arms[arm][2], arms[arm][3])
    return [obj1, obj2]


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
        pareto_distances = [
            np.sqrt((arms[pareto_arm][0] - arms[arm][0]) ** 2 + (arms[pareto_arm][2] - arms[arm][2]) ** 2) for
            pareto_arm in pareto_arms]
        return min(pareto_distances)


# RUNNING THE EXPERIMENT
def run_experiment(num_arms, num_objectives, arms, pareto_arms, weights, log=False):
    setup = {
        "Pareto UCB1": {
            "agent": PUCB1Bandit(num_arms, num_objectives, 1),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        },
        "Linear Scalarized UCB1": {
            "agent": LSUCB1Bandit(num_arms, num_objectives, weights, 1),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        },
        "Pareto Knowledge Gradient": {
            "agent": PKGBandit(num_arms, num_objectives, horizon, 3),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        },
        "Linear Scalarized Knowledge Gradient (arms)": {
            "agent": LSKGArmsBandit(num_arms, num_objectives, horizon, 3, weights),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        },
        "Linear Scalarized Knowledge Gradient (objectives)": {
            "agent": LSKGObjectivesBandit(num_arms, num_objectives, horizon, 3, weights),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        },
        "Annealing Pareto": {
            "agent": APBandit(num_arms, num_objectives, horizon, 3, 1, 0.99),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        },
        "Pareto Thompson Sampling": {
            "agent": NormalPTSBandit(num_arms, num_objectives),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        },
        "Linear Scalarized Thompson Sampling": {
            "agent": NormalLSTSBandit(num_arms, num_objectives, weights),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        }
    }

    for algorithm in setup:
        agent = setup[algorithm]["agent"]
        cumulative_pareto_regrets = setup[algorithm]["cumulative_pareto_regrets"]
        cumulative_unfairness_regrets = setup[algorithm]["cumulative_unfairness_regrets"]

        for experiment in range(num_runs):
            print(f"Experiment {experiment} for algorithm {algorithm}")
            arm_pulls = setup[algorithm]["arm_pulls"][experiment]
            agent.reset()

            for t in range(horizon):
                # Choose an arm
                arm = agent.choose_arm()
                # Pull the arm and receive the reward
                reward = pull(arm, arms)
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

            # Store the arm pulls
            setup[algorithm]["arm_pulls"][experiment] = arm_pulls

    # Plot the cumulative pareto regrets and the cumulative unfairness regrets
    plot_regrets(setup)
    plot_arm_pulls(setup, pareto_arms, horizon)


if __name__ == '__main__':
    # Run the experiment
    run_experiment(num_arms, num_objectives, arms, pareto_arms, weights)
