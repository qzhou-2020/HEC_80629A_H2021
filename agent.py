from os import path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf

from replay import Transition
from observer import HistoryObserver


def DenseNet(input_shape: tuple, nb_actions: int, hidden_layers: tuple):
    layers = [tf.keras.layers.Input(shape=input_shape)]
    layers = layers + [
        tf.keras.layers.Dense(size, activation="relu") for size in hidden_layers
    ]
    layers = layers + [tf.keras.layers.Dense(nb_actions, activation=None)]
    return tf.keras.Sequential(layers)


def Conv2dNet(input_shape, nb_actions, structure):
    # todo: wire up structure
    # this Conv2D-net is based on
    # https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    layers = [
        tf.keras.layers.Conv2D(16, (8, 8), strides=(4, 4), input_shape=input_shape, activation="relu"),
        tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu")
    ]
    layers = layers + [tf.keras.layers.Dense(nb_actions, activation=None)]
    return tf.keras.Sequential(layers)


def LstmNet(input_shape, nb_actions, structure):
    raise NotImplementedError


class DQN:

    def __init__(self, config, state_shape, nb_action, optimizer=None, loss_function=None):
        self.nb_action = nb_action
        self.state_shape = state_shape
        self.gamma = config['gamma']
        self.min_epsilon = config['min_epsilon']
        self.epsilon = 1
        self.online_network = self._network(config)
        self.target_network = self._network(config)       
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.use_ddqn = True if config.get("config", "ddqn") == "ddqn" else False
        self.sync_target_network()

    def _network(self, config):
        if config["network"]["type"] == "dense":
            return DenseNet(self.state_shape, self.nb_action, config["network"]["hidden_layers"])
        elif config["network"]["type"] == "conv2d":
            return Conv2dNet(self.state_shape, self.nb_action, config["network"].get("structure", None))
        else:
            raise NotImplementedError
    
    @tf.function
    def learn(self, e, w):
        """update online network"""

        # parse, e is Dict[str: np.ndarray]
        states, actions, rewards, states_, dones = e["s"], e["a"], e["r"], e["s_"], e["done"]
        
        # future qa_val
        if not self.use_ddqn:
            # DQN
            qa_val_ = tf.reduce_max(self.target_network(states_), axis=1)
        else:
            # DDQN
            a_ = tf.argmax(self.online_network(states_), axis=1)
            m = tf.one_hot(a_, self.nb_action)
            qa_val_ = tf.reduce_sum(tf.multiply(self.target_network(states_), m), axis=1)

        # target 
        target = tf.cast(rewards, tf.float32) + self.gamma * qa_val_ * (1 - tf.cast(dones, tf.float32))
        
        # Q-network
        mask = tf.one_hot(actions, self.nb_action)
        with tf.GradientTape() as tape:
            tape.watch(self.online_network.trainable_variables)
            q_val = self.online_network(states)
            qa_val = tf.reduce_sum(tf.multiply(q_val, mask), axis=1)
            # pay attention to the shape of sample_weight here.
            w_ = w[np.newaxis,:] if w is not None else w
            loss = self.loss_function(target, qa_val, sample_weight=w_)
       
        # backpropagation
        grads = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_network.trainable_variables))
        
        return loss

    @tf.function
    def temporal_difference(self, e):
        # parse, e is Dict[str: np.ndarray]
        states, actions, rewards, states_, dones = e["s"], e["a"], e["r"], e["s_"], e["done"]
        # future qa_val
        qa_val_ = tf.reduce_max(self.target_network(states_), axis=1)
        target = tf.cast(rewards, tf.float32) + self.gamma * qa_val_ * (1 - tf.cast(dones, tf.float32))
        # Q-network
        mask = tf.one_hot(actions, self.nb_action)
        q_val = self.online_network(states)
        qa_val = tf.reduce_sum(tf.multiply(q_val, mask), axis=1)
        return target - qa_val

    def choose_action(self, state):
        """e-greedy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.nb_action)
        else:
            actions = self.online_network(state[np.newaxis, :], training=False)
            return tf.argmax(actions[0]).numpy()

    def update_epsilon(self, timestep, decay=1e-5):
        self.epsilon = self.min_epsilon + (1. - self.min_epsilon) * np.exp(-decay * timestep)

    def update_target_network(self, tau):
        """update target network softly every time step"""
        for t, e in zip(
            self.target_network.trainable_variables, self.online_network.trainable_variables
        ):
            t.assign(t * (1-tau) + e * tau)

    def sync_target_network(self):
        """copy online network to target network, peroidic"""
        for t, e in zip(
            self.target_network.trainable_variables, self.online_network.trainable_variables
        ):
            t.assign(e)

    def save_policy(self, filepath):
        self.online_network.save(filepath)


class Policy:

    def __init__(self, saved_model_path):
        self.online_network = tf.keras.models.load_model(saved_model_path)

    def choose_action(self, state):
        actions = self.online_network(state[np.newaxis, :], training=False)
        return tf.argmax(actions[0]).numpy()


def train(
    env, agent, buffer, num_episodes=1000, max_steps_per_episode=10000, batch_size=64,
    online_update_period=1, target_sync_period=4, log_interval=100, use_soft_update=False,
    target_update_tau=1, decay_rate=1e-5, observer=None, num_saves=0, early_stop=False, 
    saved_model_dir=None
):
    """train the agent"""

    save_num_episode = None if num_saves == 0 else num_episodes // num_saves

    # stack and time index
    frame_count = 0
    reward_history = HistoryObserver(num_episodes)

    # tf.device
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        device_name = '/cpu:0'

    # saved model path
    if saved_model_dir is None:
        saved_model_dir = path.join(path.abspath(path.dirname(__file__)), "saved_model_{}".format(env.unwrapped.spec.id))

    # training loop
    for Ei in range(num_episodes):
        s = env.reset() # current state
        episode_reward = 0
        for ts in range(1, max_steps_per_episode):
            frame_count += 1

            # get action
            agent.update_epsilon(frame_count, decay=decay_rate)
            a = agent.choose_action(s)
            # interaction
            s_, r, done, _ = env.step(a)
            # update reward
            episode_reward += r
            
            # transition
            tr = Transition(s, a, r, s_, done)
            # save transition
            if buffer.per: # Prioritized replay buffer
                td = agent.temporal_difference({
                    "s": np.array([s]),
                    "a": np.array([a]),
                    "r": np.array([r]),
                    "s_": np.array([s_]),
                    "done": np.array([done])
                }).numpy()[0]
                buffer.append(tr, td)
            else:
                buffer.append(tr)

            # learning

            if buffer.nb_frames < batch_size:
                if done: break
                continue

            if frame_count % online_update_period == 0:
                e, idx, w = buffer.sample(batch_size) # namedtuple
                if buffer.per:
                    with tf.device(device_name):
                        TDs = agent.temporal_difference(e).numpy()
                    for i, td in zip(idx, TDs):
                        buffer.update(i, td)

                with tf.device(device_name):
                    loss_value = agent.learn(e, w)
                
                if use_soft_update:
                    agent.update_target_network(target_update_tau)
            
            if frame_count % target_sync_period == 0:
                agent.sync_target_network()

            if done:
                break

            s = s_ # update current state

        reward_history.append(episode_reward)

        if observer:
            for obv in observer:
                obv.append(episode_reward)

            if Ei % log_interval == log_interval - 1:
                msg = [f"period: {Ei+1}"]
                for obv in observer:
                    msg.append(f"{obv.name} reward: {obv.result:.3f}")
                msg.append(f"epsilon: {agent.epsilon:0.4f}")
                if buffer.per:
                    msg.append(f"max priority: {buffer.max_priority:.1f}")
                logger.info(", ".join(msg))


        if save_num_episode is not None:
            if Ei % save_num_episode == 0:
                agent.save_policy(path.join(saved_model_dir, f"period_{Ei}"))

        # todo: early termination
        # terminate if the training is "converged"

    # save online network
    agent.save_policy(path.join(saved_model_dir, "final_model"))

    return reward_history.result