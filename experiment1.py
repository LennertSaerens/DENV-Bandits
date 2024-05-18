import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# EXPERIMENTAL SETUP PARAMETERS
num_runs = 100  # Number of experiments M
horizon = 30_000  # Number of time steps T

# LOADING DATA
experiment_data = pd.read_csv('results/Experiment1Extended.csv')

optimal_arms = [0, 5, 6, 8, 14, 30, 31, 32]


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
