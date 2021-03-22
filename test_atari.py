import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

import numpy as np
import gym
import tensorflow as tf

from agent import DQN, train
from replay import UniformReplayBuffer, PrioritizedReplayBuffer
from replay import Transition
from observer import AverageObserver, MaximumObserver
from wrapper import AtariWrapper


def run(
    agent_type="dqn",
    gamma=1.0,
    min_epsilon=0.001,
    learning_rate=2.5e-4,
    env_name="MsPacman-v0",
    use_wrapper=True,
    num_episodes=1000,
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
    if use_wrapper:
        # convert (210, 160, 3) to (84, 84, 1)
        env = AtariWrapper(env)

    cfg = {
        "type": agent_type,
        "network": {
            "type": "conv2d",
            "structure": None,
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

    for target_sync_period in [100]:
        logging.info(f"\ntarget_sync_period = {target_sync_period}")
        run(
            agent_type="ddqn",
            target_sync_period=200,
            use_prioritized_experience_buffer=False,
            num_episodes=1000,
            log_interval=50
        )