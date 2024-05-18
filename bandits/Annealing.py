import numpy as np


class APBandit:
    """
    Implementation of the Annealing Pareto algorithm.
    From "Multivariate Normal Distribution Based Multi-Armed Bandits Pareto Algorithm" by Saba Q. Yahyaa, Madalina M. Drugan and Bernard Manderick
    """
    def __init__(self, num_arms, num_objectives, timesteps, initialization_phases, epsilon, annealing_rate):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.timesteps = timesteps
        self.initialization_phases = initialization_phases
        self.arm_means = np.zeros((num_arms, num_objectives))
        self.arm_stds = np.zeros((num_arms, num_objectives))
        self.arm_pulls = np.zeros((num_arms, num_objectives))
        self.current_init_arm = 0
        self.current_init_phase = 0
        self.epsilon = epsilon
        self.copy_epsilon = epsilon
        self.annealing_rate = annealing_rate
        self.pareto_front = np.arange(num_arms)

    def choose_arm(self):
        """
        Choose an arm to pull based on the Annealing Pareto strategy.
        :return: The arm to pull.
        """
        self.epsilon *= self.annealing_rate
        if self.current_init_phase < self.initialization_phases:
            arm = self.current_init_arm
            self.current_init_arm += 1
            if self.current_init_arm == self.num_arms:
                self.current_init_arm = 0
                self.current_init_phase += 1
        else:
            arms_in_range = []
            for o in range(self.num_objectives):
                # get the highest mean for the objective
                max_mean = np.max(self.arm_means[:, o])
                # get the arms that are in the epsilon range of the highest mean
                arms_in_range.extend(np.where(self.arm_means[:, o] >= max_mean - self.epsilon)[0])
            arms_in_range = np.unique(arms_in_range)
            difference = np.setdiff1d(self.pareto_front, arms_in_range)
            # add the arms from difference that are not dominated by any other arm, to the arms in range
            for arm in difference:
                if not np.any(np.all(self.arm_means[arms_in_range] >= self.arm_means[arm], axis=1)):
                    arms_in_range = np.append(arms_in_range, arm)
            self.pareto_front = arms_in_range
            arm = np.random.choice(self.pareto_front)
        self.arm_pulls[arm] += 1
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
        self.epsilon = self.copy_epsilon
        self.pareto_front = np.arange(self.num_arms)
        self.arm_means = np.zeros((self.num_arms, self.num_objectives))
        self.arm_stds = np.zeros((self.num_arms, self.num_objectives))
        self.arm_pulls = np.zeros((self.num_arms, self.num_objectives))
        self.current_init_arm = 0
        self.current_init_phase = 0

    def get_top_arms(self, num_arms):
        """
        Get the top arms based on the means.
        :param num_arms: The number of top arms to get.
        :return: The top arms.
        """
        return np.argsort(np.sum(self.arm_means, axis=1))[::-1][:num_arms]
