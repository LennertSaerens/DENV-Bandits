import numpy as np
import math
import random


class ParetoUCB1Bandit:
    """
    Empirical Pareto UCB1 Bandit
    From "Designing multi-objective multi-armed bandits algorithms: a study" by Madalina M. Drugan and Ann Nowe
    """

    def __init__(self, num_arms, num_objectives, kappa):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.kappa = kappa
        self.n = 0
        self.arm_means = np.zeros((num_arms, num_objectives))
        self.arm_counts = np.zeros((num_arms, num_objectives))
        self.current_init_arm = 0

    def choose_arm(self):
        """
        Choose an arm to pull based on the Upper Confidence Bound (UCB) strategy.
        :return: The arm to pull.
        """
        if np.all(self.arm_counts > 0):
            ucb_values = self.arm_means + self.kappa * np.sqrt(
                (2 * math.log(self.n * pow(self.num_objectives * self.num_arms, 1 / 4))) / self.arm_counts
            )
            is_strictly_worse = np.all(ucb_values[:, None, :] < ucb_values[None, :, :], axis=2)
            pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
            arm = random.choice(pareto_indices)
        else:
            arm = self.current_init_arm
            self.current_init_arm += 1
        self.arm_counts[arm] += 1
        self.n += 1
        return arm

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        self.arm_means[arm] += (reward - self.arm_means[arm]) / self.arm_counts[arm]

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.n = 0
        self.arm_means = np.zeros((self.num_arms, self.num_objectives))
        self.arm_counts = np.zeros((self.num_arms, self.num_objectives))
        self.current_init_arm = 0


class ScalarizedUCB1Bandit:
    """
    Scalarized UCB1 Bandit
    From "Designing multi-objective multi-armed bandits algorithms: a study" by Madalina M. Drugan and Ann Nowe
    """

    def __init__(self, num_arms, num_objectives, scalarization_functions, kappa):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.scalarization_functions = scalarization_functions
        self.num_scalarization_functions = len(scalarization_functions)
        self.kappa = kappa
        self.n = 0
        self.arm_means = np.zeros((self.num_scalarization_functions, num_arms, num_objectives))
        self.arm_counts = np.zeros((self.num_scalarization_functions, num_arms))
        self.current_init_arm = 0
        self.current_init_function = 0
        self.MRU = False  # Most Recently Used scalarization function

    def choose_arm(self):
        """
        Choose an arm to pull based on the Upper Confidence Bound (UCB) strategy.
        :return: The arm to pull.
        """
        if np.all(self.arm_counts > 0):
            # Pick a random scalarization function
            function = random.choice(range(self.num_scalarization_functions))
            scalarized_means = np.dot(self.arm_means[function], self.scalarization_functions[function])
            ucb_values = scalarized_means + self.kappa * np.sqrt(2 * math.log(self.n) / self.arm_counts[function])
            arm = np.argmax(ucb_values)
        else:
            function = self.current_init_function
            arm = self.current_init_arm
            self.current_init_arm += 1
            if self.current_init_arm == self.num_arms:
                self.current_init_arm = 0
                self.current_init_function += 1
        self.arm_counts[function][arm] += 1
        self.n += 1
        self.MRU = function
        return arm

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        self.arm_means[self.MRU][arm] += (reward - self.arm_means[self.MRU][arm]) / self.arm_counts[self.MRU][arm]

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.n = 0
        self.arm_means = np.zeros((self.num_scalarization_functions, self.num_arms, self.num_objectives))
        self.arm_counts = np.zeros((self.num_scalarization_functions, self.num_arms))
        self.current_init_arm = 0
        self.current_init_function = 0
        self.MRU = False
