import numpy as np
import random


class TTPFTSBandit:
    """
    Top Two Pareto Front Thompson Sampling Bandit

    MO adaptation of the Top Two Thompson Sampling Bandit for expansion of the best-arm identification problem to the
    multi-objective case: Pareto front identification.
    """

    def __init__(self, num_arms, num_objectives):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.alphas = np.ones((num_arms, num_objectives))
        self.betas = np.ones((num_arms, num_objectives))

    def choose_arm(self):
        """
        Find all Pareto optimal arms. With a chance of 1/2, return one of them. Otherwise, find the non-dominated arms
        in the non-Pareto optimal set and return one of them.
        :return: The arm to pull.
        """
        pareto_indices = self.get_top_arms()
        if np.random.random() < 0.5:
            return random.choice(pareto_indices)
        else:
            non_pareto_indices = np.setdiff1d(np.arange(self.num_arms), pareto_indices)
            samples = np.random.beta(self.alphas[non_pareto_indices], self.betas[non_pareto_indices])
            is_strictly_worse = np.all(samples[:, None, :] < samples[None, :, :], axis=2)
            non_dominated_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
            return random.choice(non_dominated_indices)

    def get_top_arms(self):
        """
        Get the arms that are considered to be Pareto optimal by the bandit.
        :return: The top arms.
        """
        samples = np.random.beta(self.alphas, self.betas)
        is_strictly_worse = np.all(samples[:, None, :] < samples[None, :, :], axis=2)
        pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
        return pareto_indices

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
