import numpy as np
import tensorflow as tf


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
        qa_val_ = self.target_network(states_).numpy().max(axis=1)
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




