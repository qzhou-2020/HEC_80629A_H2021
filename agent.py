from os import path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf

from replay import Transition
from observer import HistoryObserver


def DenseNet(input_shape: tuple, nb_actions: int, hidden_layers: list):
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
        tf.keras.layers.Lambda(lambda x: x / 255., input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu"),
        tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu"),
        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu")
    ]
    layers = layers + [tf.keras.layers.Dense(nb_actions, activation="linear")]
    return tf.keras.Sequential(layers)


def LstmNet(input_shape, nb_actions, structure):
    raise NotImplementedError


class AnnealingEpsilon:
    def __init__(self, epsilon=0.1, decay_rate=1e-5, **kwargs):
        self.epsilon = 1.0
        self.min_epsilon = epsilon
        self.decay_rate = decay_rate
        self.clock = 0

    def update(self):
        self.clock += 1
        self.epsilon = self.min_epsilon + (1. - self.min_epsilon) \
                     * np.exp(-self.decay_rate * self.clock)


class DQN:
    def __init__(self, config, state_shape, nb_action):
        self.nb_action = nb_action
        self.state_shape = state_shape
        self.use_ddqn = True if config["type"] == "ddqn" else False

        self.online_network = self._network(config["network"])
        self.target_network = self._network(config["network"])       
        self.sync_target_network()
        
        self.gamma = config['gamma']
        
        if config["explore"]["type"] == "annealing_epsilon":
            self.explore_policy = AnnealingEpsilon(**config["explore"])
        else:
            raise NotImplementedError
        
        if config["optimizer"]["type"] == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config["optimizer"]["learning_rate"])
        else:
            raise NotImplementedError
        self.optimizer = optimizer

        if config["optimizer"]["loss"] == "mse":
            loss_function = tf.keras.losses.MeanSquaredError()
        elif config["optimizer"]["loss"] == "mae":
            loss_function = tf.keras.losses.MeanAbsoluteError()
        elif config["optimizer"]["loss"] == "huber":
            loss_function = tf.keras.losses.Huber()
        else:
            raise NotImplementedError
        self.loss_function = loss_function

    def _network(self, config):
        if config["type"] == "dense":
            return DenseNet(self.state_shape, self.nb_action, config["hidden_layer_size"])
        elif config["type"] == "conv2d":
            return Conv2dNet(self.state_shape, self.nb_action, config.get("conv2d_structure", None))
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
        self.explore_policy.update()
        if np.random.random() < self.explore_policy.epsilon:
            return np.random.choice(self.nb_action)
        else:
            actions = self.online_network(state[np.newaxis, :], training=False)
            return tf.argmax(actions[0]).numpy()

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
    env, agent, buffer, 
    mode = "step",
    length = 1000,
    batch_size = 32,
    max_steps_per_episode = 1000,
    warm_up = 1000,
    online_update_period = 1,
    target_sync_period = 1,
    use_soft_target_update = False,
    target_soft_update_tau = 1.0,
    early_stop = False,
    log_interval = 100,
    save_final_model = True,
    save_intermediate_model_at = [],
    save_model_to = None,
    observer = None,
    **kwargs
):
    """train the agent"""
    # tf.device
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        device_name = '/cpu:0'

    logger.info(f"use DDQN: {agent.use_ddqn}")
    logger.info(f"use PER: {buffer.per}")
    logger.info(f"use device: {device_name}")

    # if save intermediate model
    save_intermediate_model = False if len(save_intermediate_model_at) == 0 else True

    # stack and time index
    frame_count = 0
    episode_count = 0
    new_episode = False
    reward_history = []

    # saved model path
    if save_model_to is None:
        save_model_to = path.join(path.abspath(path.dirname(__file__)), "saved_model_{}".format(env.unwrapped.spec.id))

    # training loop
    episode_reward = 0.
    s = env.reset() # current state
    for index in range(length):
        for ts in range(1, max_steps_per_episode):
            frame_count += 1
            # get action
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

            if buffer.nb_frames > warm_up:
                # learning from experience
                if frame_count % online_update_period == 0:
                    e, idx, w = buffer.sample(batch_size) # namedtuple
                    if buffer.per:
                        with tf.device(device_name):
                            TDs = agent.temporal_difference(e).numpy()
                        for i, td in zip(idx, TDs):
                            buffer.update(i, td)

                    with tf.device(device_name):
                        loss_value = agent.learn(e, w)
                    
                    if use_soft_target_update:
                        agent.update_target_network(target_soft_update_tau)

                # sync target network
                if frame_count % target_sync_period == 0:
                    agent.sync_target_network() 
            
            if done or ts == max_steps_per_episode-1:
                episode_count += 1
                new_episode = True
                break

            s = s_

            if mode == "step":
                break

            # end of max_step_per_episode loop
        if new_episode:
            reward_history.append(episode_reward)
            if observer:
                for obv in observer:
                    obv.append(episode_reward)
            # reset
            new_episode = False
            episode_reward = 0
            s = env.reset()

        if index % log_interval == log_interval - 1:
            msg = [f"frame: {frame_count}", f"episode: {episode_count}"]
            for obv in observer:
                msg.append(f"{obv.name} reward: {obv.result:.3f}")
            msg.append(f"epsilon: {agent.explore_policy.epsilon:0.4f}")
            if buffer.per:
                msg.append(f"max priority: {buffer.max_priority:.1f}")
            logger.info(", ".join(msg))

        if save_intermediate_model:
            if index+1 in save_intermediate_model_at:
                agent.save_policy(path.join(save_model_to, f"step_{index+1}"))

        # terminate if the training is "converged"
        if early_stop:
        # todo: early termination
            pass

    # save online network
    agent.save_policy(path.join(save_model_to, "final_model"))

    return reward_history