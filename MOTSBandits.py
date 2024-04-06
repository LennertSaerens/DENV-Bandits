import numpy as np
import random
import math


class ParetoThompsonSamplingBandit:
    """
    Pareto Thompson Sampling Bandit (PTS)
    From "Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem" by Saba Yahyaa and Bernard Manderick
    """

    def __init__(self, num_arms, num_objectives):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.alphas = np.ones((num_arms, num_objectives))
        self.betas = np.ones((num_arms, num_objectives))

    def choose_arm(self):
        """
        Find all Pareto optimal arms and choose one uniformly at random.
        :return: The arm to pull.
        """
        pareto_arms = []
        samples = np.random.beta(self.alphas, self.betas)
        is_strictly_worse = np.all(samples[:, None, :] < samples[None, :, :], axis=2)
        pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
        # Compare each of the arms with all the other arms, except itself. An arm is appended to the pareto_arms list if
        # its samples for each of the objectives are greater than or equal to the samples of all the other arms.
        # for arm in range(self.num_arms):
        #     is_pareto = True
        #     for other_arm in range(self.num_arms):
        #         if arm == other_arm:
        #             continue
        #         if np.all(samples[arm] < samples[other_arm]):
        #             is_pareto = False
        #             break
        #     if is_pareto:
        #         pareto_arms.append(arm)
        # Choose an arm uniformly at random from the list of pareto optimal arms
        return random.choice(pareto_indices)

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        for o in range(self.num_objectives):
            self.alphas[arm][o] += reward[o]
            self.betas[arm][o] += 1 - reward[o]

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.alphas = np.ones((self.num_arms, self.num_objectives))
        self.betas = np.ones((self.num_arms, self.num_objectives))


class LinearScalarizedThompsonSamplingBandit:
    """
    Linear Scalarized Thompson Sampling Bandit (LSF-TS)
    From "Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem" by Saba Yahyaa and Bernard Manderick
    """

    def __init__(self, num_arms, num_objectives, scalarization_functions):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.scalarization_functions = scalarization_functions
        self.num_scalarization_functions = len(scalarization_functions)
        self.alphas = np.ones((self.num_scalarization_functions, num_arms, num_objectives))
        self.betas = np.ones((self.num_scalarization_functions, num_arms, num_objectives))
        self.MRU = False  # Most Recently Used scalarization function

    def choose_arm(self):
        """
        Find the Pareto optimal arm by sampling from the beta distribution and using the scalarization functions.
        :return: The arm to pull.
        """
        samples = np.random.beta(self.alphas, self.betas)
        # pick a random scalarization function
        scalarization_function = random.randint(0, self.num_scalarization_functions - 1)
        # Calculate the scalarized values for each arm
        scalarized_values = np.dot(samples[scalarization_function],
                                   self.scalarization_functions[scalarization_function])
        # Set the most recently used scalarization function
        self.MRU = scalarization_function
        # Find the arm with the highest scalarized value
        return np.argmax(scalarized_values)

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        for o in range(self.num_objectives):
            self.alphas[self.MRU][arm][o] += reward[o]
            self.betas[self.MRU][arm][o] += 1 - reward[o]

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.alphas = np.ones((self.num_scalarization_functions, self.num_arms, self.num_objectives))
        self.betas = np.ones((self.num_scalarization_functions, self.num_arms, self.num_objectives))


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
