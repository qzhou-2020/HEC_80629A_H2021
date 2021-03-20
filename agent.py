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
    raise NotImplementedError


def LstmNet(input_shape, nb_actions, structure):
    raise NotImplementedError


class DQN:

    def __init__(self, config, state_shape, nb_action, optimizer=None, loss_function=None):
        self.nb_action = nb_action
        self.state_shape = state_shape
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.online_network = self._network(config)
        self.target_network = self._network(config)       
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.use_ddqn = True if config.get("config", "ddqn") == "ddqn" else False
        self.sync_target_network()

    def _network(self, config):
        if config["network"]["type"] == "dense":
            return DenseNet(self.state_shape, self.nb_action, config["network"]["hidden_layers"])
        else:
            raise NotImplementedError
    
    def learn(self, e, w):
        """update online network"""

        # parse, e is Dict[str: np.ndarray]
        states, actions, rewards, states_, dones = e["s"], e["a"], e["r"], e["s_"], e["done"]
        
        # future qa_val
        if not self.use_ddqn:
            # DQN target
            qa_val_ = self.target_network(states_).numpy().max(axis=1)
            target = rewards + self.gamma * qa_val_ * (1 - dones)
        else:
            # DDQN target
            a_ = self.online_network(states_).numpy().argmax(axis=1)
            m = tf.one_hot(a_, self.nb_action).numpy()
            qa_val_ = np.sum(self.target_network(states_).numpy() * m, axis=1)
            target = rewards + self.gamma * qa_val_ * (1 - dones)
        
        # Q-network
        mask = tf.one_hot(actions, self.nb_action)
        with tf.GradientTape() as tape:
            tape.watch(self.online_network.trainable_variables)
            q_val = self.online_network(states)
            qa_val = tf.reduce_sum(tf.multiply(q_val, mask), axis=1)
            # pay attention to the shape of sample_weight here.
            loss = self.loss_function(target, qa_val, sample_weight=w[np.newaxis,:])
       
        # backpropagation
        grads = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_network.trainable_variables))
        
        return loss

    def temporal_difference(self, e):
        # parse, e is Dict[str: np.ndarray]
        states, actions, rewards, states_, dones = e["s"], e["a"], e["r"], e["s_"], e["done"]
        # future qa_val
        qa_val_ = self.target_network(states_).numpy().max(axis=1)
        target = rewards + self.gamma * qa_val_ * (1 - dones)
        # Q-network
        mask = tf.one_hot(actions, self.nb_action)
        q_val = self.online_network(states)
        qa_val = tf.reduce_sum(tf.multiply(q_val, mask), axis=1).numpy()
        return target - qa_val

    def choose_action(self, state):
        """e-greedy"""
        if np.random.random() < self.epsilon:
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


def train(
    env, agent, buffer, num_episodes=1000, max_steps_per_episode=10000, batch_size=64,
    online_update_period=1, target_sync_period=4, log_interval=100, use_soft_update=False,
    target_update_tau=1, observer=None
):
    """train the agent"""

    frame_count = 0

    reward_history = HistoryObserver(num_episodes)

    for Ei in range(num_episodes):
        s = env.reset() # current state
        episode_reward = 0
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
                })[0]
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
                    TDs = agent.temporal_difference(e)
                    for i, td in zip(idx, TDs):
                        buffer.update(i, td)
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
                print(", ".join(msg))

    return reward_history.result