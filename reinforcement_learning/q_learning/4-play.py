#!/usr/bin/env python3
"""
Play module.
"""


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Args:
        env: The FrozenLakeEnv instance.
        Q: A numpy.ndarray containing the Q-table.
        max_steps: The maximum number of steps in the episode.

    Returns:
        The total rewards for the episode.
    """
    state = env.reset()
    total_reward = 0
    print(env.render())  # Display the initial state

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        state = new_state
        print(env.render()) # Display each state

        if done:
            break
    return total_reward
