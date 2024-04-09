import numpy as np
import random
from scipy.stats import norm


def x(zeta):
    Phi = norm.cdf
    phi = norm.pdf
    return zeta * Phi(zeta) + phi(zeta)


class ParetoKnowledgeGradientBandit:
    """
    Implementation of the Pareto Knowledge Gradient algorithm.
    From "Knowledge Gradient for Multi-objective Multi-armed Bandit Algorithms"
    by Saba Q. Yahyaa, Madalina M. Drugan and Bernard Manderick
    """

    def __init__(self, num_arms, num_objectives, timesteps, initialization_phases):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.timesteps = timesteps
        self.initialization_phases = initialization_phases
        self.t = 0
        self.arm_means = np.zeros((num_arms, num_objectives))
        self.arm_stds = np.zeros((num_arms, num_objectives))
        self.arm_pulls = np.zeros((num_arms, num_objectives))
        self.current_init_arm = 0
        self.current_init_phase = 0

    def choose_arm(self):
        """
        Choose an arm to pull based on the Knowledge Gradient strategy.
        :return: The arm to pull.
        """
        if self.current_init_phase < self.initialization_phases:
            arm = self.current_init_arm
            self.current_init_arm += 1
            if self.current_init_arm == self.num_arms:
                self.current_init_arm = 0
                self.current_init_phase += 1
        else:
            kg_values = self.arm_means + (
                    (self.timesteps - self.t) *
                    (self.num_arms * self.num_objectives) *
                    (self.arm_stds / np.sqrt(self.arm_pulls)) * x(-np.abs((self.arm_means - np.max(self.arm_means)) / (self.arm_stds / np.sqrt(self.arm_pulls))))
            )
            is_strictly_worse = np.all(kg_values[:, None, :] < kg_values[None, :, :], axis=2)
            pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
            arm = random.choice(pareto_indices)
        self.arm_pulls[arm] += 1
        self.t += 1
        return arm

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        self.arm_means[arm] += (reward - self.arm_means[arm]) / self.arm_pulls[arm]
        self.arm_stds[arm] = np.sqrt(np.sum((reward - self.arm_means[arm]) ** 2) / self.arm_pulls[arm])

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.t = 0
        self.arm_means = np.zeros((self.num_arms, self.num_objectives))
        self.arm_stds = np.zeros((self.num_arms, self.num_objectives))
        self.arm_pulls = np.zeros((self.num_arms, self.num_objectives))
        self.current_init_arm = 0
        self.current_init_phase = 0
