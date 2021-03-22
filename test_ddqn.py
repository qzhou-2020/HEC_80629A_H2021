import logging
logging.basicConfig(level=logging.DEBUG, filename="test_dqn_per.log")

import numpy as np
import gym
import tensorflow as tf

from agent import DQN
from agent import train
from replay import UniformReplayBuffer, PrioritizedReplayBuffer
from replay import Transition
from observer import AverageObserver, MaximumObserver


def run(
    agent_type="dqn",
    hidden_layer_size=32,
    gamma=1.0,
    min_epsilon=0.001,
    learning_rate=2.5e-4,
    env_name="CartPole-v0",
    num_episodes=3000,
    log_interval=100,
    replay_buffer_capacity=10**5,
    use_prioritized_experience_buffer=False,
    max_steps_per_episode = 10000,
    batch_size = 32,
    use_soft_update = False,
    online_update_period = 1,
    target_update_tau = 1,
    target_sync_period = 100,
):
    env = gym.make(env_name)

    cfg = {
        "type": agent_type,
        "network": {
            "type": "dense",
            "hidden_layers": (hidden_layer_size, hidden_layer_size),
        },
        "gamma": gamma,
        "min_epsilon": min_epsilon
    }
    agent = DQN(
        cfg, 
        env.observation_space.shape, 
        env.action_space.n,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss_function=tf.keras.losses.MeanSquaredError(),
    )

    if use_prioritized_experience_buffer:
        buffer = PrioritizedReplayBuffer(
            size=replay_buffer_capacity, 
            alpha=0.6, 
            anneal_alpha_rate=1e-5, 
            anneal_beta_rate=1e-5
        )
    else:
        buffer = UniformReplayBuffer(size=replay_buffer_capacity)

    observer = [
        AverageObserver(log_interval), 
        MaximumObserver(log_interval)
    ]

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

if __name__ == "__main__":

    logging.info("This is to test target sync period effect.")

    for target_sync_period in [100, 200]:
        logging.info(f"\ntarget_sync_period = {target_sync_period}")
        run(
            agent_type="ddqn",
            target_sync_period=200,
            use_prioritized_experience_buffer=True
            num_episodes=5000,
            log_interval=50
        )