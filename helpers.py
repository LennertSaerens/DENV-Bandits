import numpy as np


#
# Functions for creating arms
#

def create_arm(num_objectives):
    """
    Create an arm with success rates for each of the objectives. The success rates are sampled from a normal
    distribution with mean 0.5 and standard deviation 0.1.
    :param num_objectives: The number of objectives.
    :return: The arm.
    """
    return [np.random.normal(0.5, 0.05) for _ in range(num_objectives)]


def is_pareto_optimal(arm, other_arms):
    """
    Check if the arm is Pareto optimal with respect to the other arms.
    :param arm: The arm.
    :param other_arms: The other arms.
    :return: True if the arm is Pareto optimal, False otherwise.
    """
    for other_arm in other_arms:
        if np.all(np.array(arm) <= np.array(other_arm)):
            return False
    return True


def create_optimal_arm(num_objectives, other_arms):
    """
    Create an arm with success rates for each of the objective that is Pareto optimal with respect to the other arms.
    :param num_objectives: The number of objectives.
    :param other_arms: The other arms.
    :return: The arm.
    """
    arm = create_arm(num_objectives)
    while not is_pareto_optimal(arm, other_arms):
        arm = create_arm(num_objectives)
    return arm


def create_suboptimal_arm(num_objectives, other_arms):
    """
    Create an arm with success rates for each of the objective that is not Pareto optimal with respect to the other arms.
    :param num_objectives: The number of objectives.
    :param other_arms: The other arms.
    :return: The arm.
    """
    arm = create_arm(num_objectives)
    while is_pareto_optimal(arm, other_arms):
        arm = create_arm(num_objectives)
    return arm


def create_arms(num_optimal, num_suboptimal, num_objectives):
    """
    Create a set of arms with a specified number of optimal and suboptimal arms.
    :param num_optimal: The number of optimal arms.
    :param num_suboptimal: The number of suboptimal arms.
    :param num_objectives: The number of objectives.
    :return: The set of arms.
    """
    assert num_optimal + num_suboptimal > 1
    arms = []
    # Create num_optimal optimal arms
    for _ in range(num_optimal):
        arm = create_optimal_arm(num_objectives, arms)
        arms.append(arm)
    # Create num_suboptimal suboptimal arms
    for _ in range(num_suboptimal):
        arm = create_suboptimal_arm(num_objectives, arms)
        arms.append(arm)
    # Scramble the arms
    np.random.shuffle(arms)
    return arms


def num_optimal_arms(arms):
    """
    Calculate the number of optimal arms in the set of arms.
    :param arms: The set of arms.
    :return: The number of optimal arms.
    """
    optimal = 0
    for arm in arms:
        if is_pareto_optimal(arm, arms):
            optimal += 1
    return optimal


#
# Functions for creating weights
#

def create_random_weights(num_objectives, num_weights):
    """
    Create a set of weights for the linear scalarization function.
    :param num_objectives: The number of objectives.
    :param num_weights: The number of weights.
    :return: The set of weights.
    """
    return [np.random.random(num_objectives) for _ in range(num_weights)]
