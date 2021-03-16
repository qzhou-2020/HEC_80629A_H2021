import numpy as np
import gym
import tensorflow as tf

from agent import DQN
from replay import UniformReplayBuffer, PrioritizedReplayBuffer
from replay import Transition


class Observer(object):
    def __init__(self, size=10):
        self.size = size
        self.buffer = [0 for _ in range(size)]
        self.index = 0
        self.full = False
    
    def append(self, reward):
        self.buffer[self.index] = reward
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0

    @property
    def average(self):
        nb = self.size if self.full else self.index
        return sum(self.buffer) / nb

    @property
    def max(self):
        return max(self.buffer)
    
    @property
    def min(self):
        return min(self.buffer)


env_name = "CartPole-v1"
num_episodes = 5000
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
    loss_function=tf.keras.losses.MeanSquaredError()
)

# buffer
if use_prioritized_experience_buffer:
    buffer = PrioritizedReplayBuffer(size=replay_buffer_capacity, alpha=0.6)
else:
    buffer = UniformReplayBuffer(size=replay_buffer_capacity)

# observer
observer = Observer(size=10)

frame_count = 0

for Ei in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    for ts in range(1, max_steps_per_episode):
        frame_count += 1
        
        # get action
        action = agent.choose_action(state)
        
        # interact with env
        state_, reward, done, info = env.step(action)
        episode_reward += reward
        
        # save to replay buffer
        transition = Transition(state, action, reward, state_, done)
        if use_prioritized_experience_buffer:
            td = agent.temporal_difference({
                "s": np.array([state]),
                "a": np.array([action]),
                "r": np.array([reward]),
                "s_": np.array([state_]),
                "done": np.array([done])
            })[0]
            buffer.append(transition, td)
        else:
            buffer.append(transition)

        # skip if available replays are too few
        if buffer.nb_frames < batch_size:
            if done: break
            continue
        
        # learning
        if frame_count % online_update_period == 0:
            experiences, indices, is_weight = buffer.sample(batch_size) # namedtuple
            
            if use_prioritized_experience_buffer:
                TDs = agent.temporal_difference(experiences)
                for idx, td in zip(indices, TDs):
                    buffer.update(idx, td)

            loss_value = agent.learn(experiences, is_weight)
            if use_soft_update:
                agent.update_target_network(target_update_tau)

        if frame_count % target_sync_period == 0:
            agent.sync_target_network()

        if done:
            break

        # update state and move to the next timestep
        state = state_

    observer.append(episode_reward)

    if Ei % log_interval == log_interval-1:
        print("period: {}, average episodic reward: {:.3f}, max reward: {:.3f}".format(
            Ei+1, observer.average, observer.max))
