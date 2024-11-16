#!/usr/bin/env python3
"""
Initialize the q-table.
"""

def q_init(env):
    """Initializes the Q-table.

    Args:
        env: The FrozenLakeEnv instance.

    Returns:
        The Q-table as a numpy.ndarray of zeros.
    """
    return np.zeros([env.observation_space.n, env.action_space.n])
