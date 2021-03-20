import numpy as np
import gym
import tensorflow as tf

from agent import DQN
from agent import train
from replay import UniformReplayBuffer, PrioritizedReplayBuffer
from replay import Transition
from observer import AverageObserver, MaximumObserver


env_name = "CartPole-v1"
num_episodes = 200
log_interval = 50
learning_rate = 5e-4
replay_buffer_capacity = 10000
use_prioritized_experience_buffer = True
max_steps_per_episode = 10000
batch_size = 64
use_soft_update = False
online_update_period = 1
target_update_tau = 1
target_sync_period = 1
gamma = 1
epsilon = 0.1

config = {
    "type": "dqn",
    "network": {
        "type": "dense",
        "hidden_layers": (32, 32),
    },
    "gamma": gamma,
    "epsilon": epsilon,
}

# env
env = gym.make(env_name)

# agent
agent = DQN(
    config, 
    env.observation_space.shape, 
    env.action_space.n,
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss_function=tf.keras.losses.MeanSquaredError(),
)

# buffer
if use_prioritized_experience_buffer:
    buffer = PrioritizedReplayBuffer(size=replay_buffer_capacity, alpha=0.6)
else:
    buffer = UniformReplayBuffer(size=replay_buffer_capacity)

# observer
observer = [AverageObserver(10), MaximumObserver(10)]

train(
    env, agent, buffer, 
    num_episodes=num_episodes, 
    max_steps_per_episode=max_steps_per_episode,
    batch_size=batch_size,
    online_update_period=online_update_period,
    target_sync_period=target_sync_period,
    log_interval=log_interval,
    use_soft_update=use_soft_update,
    target_update_tau=target_update_tau,
    observer=observer
)
