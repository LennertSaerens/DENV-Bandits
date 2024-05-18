import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bandits.Annealing import APBandit
from bandits.KnowledgeGradient import PKGBandit, LSKGArmsBandit, LSKGObjectivesBandit
from bandits.ThompsonSampling import NormalPTSBandit, NormalLSTSBandit
from bandits.UCB import PUCB1Bandit, LSUCB1Bandit
from main import calculate_unfairness_regret
from plotting import plot_regrets, plot_arm_pulls

# EXPERIMENTAL SETUP PARAMETERS
num_runs = 10  # Number of experiments M
horizon = 1000  # Number of time steps T

# LOADING DATA
experiment_data = pd.read_csv('results/Experiment1Extended.csv')

optimal_arms = [0, 5, 6, 8, 14, 30, 31, 32]
medical_error = 1000
monetary_error = 0.1
num_objectives = 2


# VISUALIZATION OF THE DATAFRAME
def plot_vaccination_data(df, annotate=False, connect_optimal_arms=False):
    plt.rcParams.update({'font.size': 13})
    plt.rcParams.update({'font.family': 'serif'})
    # Define colors for groups of points
    colors = ['black', 'indianred', 'lightcoral', 'moccasin', 'palegoldenrod', 'lemonchiffon', 'palegreen', 'lightcyan',
              'paleturquoise', 'darkseagreen', 'lightskyblue', "palevioletred", "pink", "lavenderblush"]
    color_index = 0
    # Get the labels for the legend from the 'Description' column
    labels = df['Description'].unique()
    # Delete the NAN label
    labels = labels[~pd.isnull(labels)]
    legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]

    # Plot the first point in black
    plt.scatter(df['Medical Burden'].iloc[0], df['Monetary Cost'].iloc[0], color="black")

    # Plot the rest of the points in groups, changing colors every 4 points
    for i in range(1, len(df)):
        if i % 4 == 1:
            color_index += 1
        # annotate the points with their index if the annotate flag is set
        if annotate:
            plt.annotate(i, (df['Medical Burden'].iloc[i], df['Monetary Cost'].iloc[i]))
        plt.scatter(df['Medical Burden'].iloc[i], df['Monetary Cost'].iloc[i], color=colors[color_index])

    # Connect each optimal arm with the next one if the connect_optimal_arms flag is set
    if connect_optimal_arms:
        for i in range(len(optimal_arms) - 1):
            plt.plot([df['Medical Burden'].iloc[optimal_arms[i]], df['Medical Burden'].iloc[optimal_arms[i + 1]]],
                     [df['Monetary Cost'].iloc[optimal_arms[i]], df['Monetary Cost'].iloc[optimal_arms[i + 1]]],
                     color='black', linestyle='dotted')

    plt.xlabel('Medical Burden')
    plt.ylabel('Monetary Cost')
    plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    plt.show()
    # Reset font size and family
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'font.family': 'sans-serif'})


# Visualize the untransformed data
plot_vaccination_data(experiment_data, annotate=True, connect_optimal_arms=True)

# TRANSFORMING THE DATA

# Divide monetary cost by 100
experiment_data['Monetary Cost'] = experiment_data['Monetary Cost'] / 89

# Set 0 monetary cost for no vaccination to 1 to avoid division by 0
experiment_data.loc[0, 'Monetary Cost'] = 1

# Invert the values of the 'Medical Burden' column to go from minimization to maximization
experiment_data['Medical Burden'] = 1 / experiment_data['Medical Burden']

# Invert the values of the 'Monetary Cost' column to go from minimization to maximization
experiment_data['Monetary Cost'] = 1 / experiment_data['Monetary Cost']

# Visualize the transformed data
plot_vaccination_data(experiment_data, annotate=True, connect_optimal_arms=True)

arms = [(experiment_data['Medical Burden'][i], experiment_data['Monetary Cost'][i]) for i in
        range(len(experiment_data))]
num_arms = len(arms)
pareto_arms = [0, 5, 6, 8, 14, 30, 31, 32]
weights = [(x, 1 - x) for x in np.linspace(0, 1, 11)]


def pull(arm, arms):
    obj1 = np.random.normal(arms[arm][0], medical_error)
    obj2 = np.random.normal(arms[arm][1], monetary_error)
    return [obj1, obj2]


def calculate_pareto_regret(arm, arms, pareto_arms):
    if arm in pareto_arms:
        return 0
    else:
        pareto_distances = [
            np.sqrt((arms[pareto_arm][0] - arms[arm][0]) ** 2 + (arms[pareto_arm][1] - arms[arm][1]) ** 2) for
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
