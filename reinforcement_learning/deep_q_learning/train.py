#!/usr/bin/env python3
"""
training module.
"""


import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Create the Atari Breakout environment
env = gym.make('Breakout-v0')
nb_actions = env.action_space.n

# Build a simple policy network
def build_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

model = build_model((1,) + env.observation_space.shape, nb_actions)

# Define memory and policy
memory = SequentialMemory(limit=1000000, window_length=1)
policy = EpsGreedyQPolicy()

# Create the DQNAgent
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy,
               nb_steps_warmup=500, target_model_update=1e-2)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Save the policy network
dqn.save_weights('policy.h5', overwrite=True)
