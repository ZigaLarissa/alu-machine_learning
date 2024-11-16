#!/usr/bin/env python3
"""
Epsion Greedy defined.
"""

def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q: A numpy.ndarray containing the q-table.
        state: The current state.
        epsilon: The epsilon to use for the calculation.

    Returns:
        The next action index.
    """
    if np.random.uniform(0, 1) < epsilon:
        # Explore
        action = np.random.randint(0, Q.shape[1])
    else:
        # Exploit
        action = np.argmax(Q[state, :])
    return action
