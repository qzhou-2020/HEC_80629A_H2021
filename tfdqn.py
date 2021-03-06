""" DQN learner using tf-agents modules"""

from os import path
import numpy as np
import tensorflow as tf

# env
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_atari
from tf_agents.environments import suite_mujoco
# agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
# replay buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
# driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import dynamic_step_driver
# misc
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver


tf.compat.v1.enable_v2_behavior()


config = {
    "env": {
        "lib": "gym", # gym, atari, or mujoco
        "name": "LunarLander-v2", # env name.
    },
    "agent": {
        "type": "dqn", # dqn or ddqn
        "network": {
            "type": "dense", # dense, conv, or lstm
            "structure": [
                (128, {"activation": "relu"}), # hidden layer, size 128, activation = relu
                (128, {"activation": "relu"}), # hidden layer, size 128, activation = relu
            ],
        },
        "optimizer": {
            "optimizer": tf.keras.optimizers.Adam, # keras optimizer,
            "learning_rate": 5e-4, # learning_rate
            "loss_fn": common.element_wise_squared_loss, # tf-agents loss function
        },
        "target": {
            "soft": 1e-3, # q_target_network soft update factor (every step)
            "period": 4, # q_target_net synchronize period.
        },
        "gamma": 1.0,
    },
    "replay_buffer": {
        "type": "uniform", # uniform or prioritized (tf_agents.replay_buffer doesn't have a prioritized buffer)
        "batch_size": 64, # size of experiences fetched from replay buffer
        "capacity": int(1e5), # buffer max size
    },
    "driver": {
        "type": "step", # episode or step
        "observers": [
            tf_metrics.AverageEpisodeLengthMetric, # observe average episode length
            tf_metrics.MaxReturnMetric # observer max return
        ],
        "length": 1, # number of loops in a single driver.run()
    },
    "saver": {
        "checkpoint": True,
        "nb_checkpoint": 1,
        "policy_saver": True,
    }
}
    

TMPDIR = path.abspath(path.dirname(__file__))



class TFDqnAgent:

    """wrapper of tf_agent DqnAgent and DdqnAgent, together with env, buffer, driver."""

    def __init__(self, config: dict):
        self.env = self._env(config["env"])
        self.agent = self._agent(config["agent"])
        self.replay = self._replay(config["replay_buffer"])
        self.replay_iter = self._replay_iter(config["replay_buffer"]["batch_size"])
        self.observers = self._observers(config["driver"]["observers"])
        self.driver = self._driver(config["driver"])

        # policy saver
        if config["saver"].get("policy_saver", False):
            self.policy_saver = policy_saver.PolicySaver(self.agent.policy)
        else:
            self.policy_saver = None

        # initialization
        self.agent.initialize()
        self.reset()

    def train(self, nb_iterations=100, log_interval=10):
        avg_reward_history = []
        time_step, policy_state = self.time_step, self.policy_state
        for t in range(nb_iterations):

            while not time_step.is_last().numpy()[0]:
                # run simulation
                time_step, policy_state = self.driver.run(time_step, policy_state)
            
                # learning
                if self.replay.num_frames().numpy() < 64:
                    continue

                experiences, _ = next(self.replay_iter)
                loss = self.agent.train(experiences).loss

            avg_reward_history.append(self.observers[0].result().numpy())

            # cmd display
            if t % log_interval == 0 or t == nb_iterations-1:
                print("episode: {:4d}".format(t), end="")
                for obv in self.observers:
                    print(", {}: {:.2f}".format(obv.name, obv.result().numpy()), end="")
                print("")

            time_step = self.env.reset()

        # save for continous training.
        self.time_step, self.policy_state = time_step, policy_state

        # save policy
        if self.policy_saver:
            self.policy_saver.save(path.join(TMPDIR, "policy"))

        return avg_reward_history
        
    def reset(self):
        """clear the buffer and train from start"""
        self.agent.train_step_counter.assign(tf.Variable(0))
        self.time_step = self.env.reset()
        self.policy_state = None
        self.replay.clear()

    def _env(self, cfg):
        """create a tf_env"""
        if cfg["lib"] == "gym":
            return tf_py_environment.TFPyEnvironment(suite_gym.load(cfg["name"]))
        else:
            raise NotImplementedError

    def _agent(self, cfg):
        """return a TFDqnAgent"""
        net = self._net(cfg["network"])
        optimizer = cfg["optimizer"]["optimizer"](
            learning_rate=cfg["optimizer"]["learning_rate"]
        )
        loss_fn = cfg["optimizer"]["loss_fn"]
        if cfg["type"].lower() == "dqn":
            return dqn_agent.DqnAgent(
                self.env.time_step_spec(),
                self.env.action_spec(),
                q_network=net,
                optimizer=optimizer,
                td_errors_loss_fn=loss_fn,
                target_update_tau=cfg["target"]["soft"],
                target_update_period=cfg["target"]["period"],
                gamma=cfg["gamma"],
                train_step_counter=tf.Variable(0)
            )
        elif cfg["type"].lower() == "ddqn":
            return dqn_agent.DdqnAgent(
                self.env.time_step_spec(),
                self.env.action_spec(),
                q_network=net,
                optimizer=optimizer,
                td_errors_loss_fn=loss_fn,
                target_update_tau=cfg["target"]["soft"],
                target_update_period=cfg["target"]["period"],
                gamma=cfg["gamma"],
                train_step_counter=tf.Variable(0)
            )
        else:
            raise ValueError("Unknown type of agent! Input type: {}".format(cfg["type"]))

    def _replay(self, cfg):
        """return replay buffer"""
        if cfg["type"] == "uniform":
            return tf_uniform_replay_buffer.TFUniformReplayBuffer(
                self.agent.collect_data_spec,
                batch_size=self.env.batch_size,
                max_length=cfg["capacity"]
            )
        else:
            raise NotImplementedError

    def _replay_iter(self, batch_size, nb_parallel=1, nb_prefetch=1):
        """return iterator to access to replay buffer"""
        dataset = self.replay.as_dataset(
            num_parallel_calls=nb_parallel,
            sample_batch_size=batch_size,
            num_steps=2,
            single_deterministic_pass=False
        ).prefetch(nb_prefetch)
        return iter(dataset)

    def _observers(self, obvs):
        """instantiate observers"""
        observers = [tf_metrics.AverageReturnMetric()] + [obv() for obv in obvs]
        return observers

    def _driver(self, cfg):
        """return a driver"""
        observers = [self.replay.add_batch] + self.observers
        if cfg["type"] == "episode":
            return dynamic_episode_driver.DynamicEpisodeDriver(
                self.env,
                self.agent.collect_policy,
                observers=observers,
                num_episodes=cfg["length"]
            )
        elif cfg["type"] == "step":
            return dynamic_step_driver.DynamicStepDriver(
                self.env,
                self.agent.collect_policy,
                observers=observers,
                num_steps=cfg["length"]
            )
        else:
            raise ValueError("Unknown type of driver! Input is {}".format(cfg["type"]))

    def _net(self, cfg):
        """return a q-network"""
        if cfg["type"].lower() == "dense":
            return self._dense_net(cfg["structure"])
        elif cfg["type"].lower() == "conv":
            return self._conv_net(cfg["structure"])
        elif cfg["type"].lower() == "lstm":
            return self._lstm_net(cfg["structure"])
        else:
            raise ValueError("Unknown type of network! Input is {}".format(cfg["type"]))

    def _dense_net(self, structure):
        """Dense-layered sequential network"""
        nb_actions = self._nb_actions()
        layers = [tf.keras.layers.Dense(size, **keys) for size, keys in structure]
        layers.append(tf.keras.layers.Dense(nb_actions, activation=None))
        return sequential.Sequential(layers)

    def _conv_net(self, structure):
        """Conv2D sequential network"""
        raise NotImplementedError

    def _lstm_net(self, structure):
        """LSTM network"""
        raise NotImplementedError

    def _nb_actions(self):
        """return number of actions"""
        action_tensor_spec = tensor_spec.from_spec(self.env.action_spec())
        return action_tensor_spec.maximum - action_tensor_spec.minimum + 1


def test():
    import copy
    _cfg = copy.deepcopy(config)
    _cfg["driver"]["type"] = "step"
    _cfg["saver"]["policy_saver"] = False
    agent = TFDqnAgent(_cfg)
    rewards = agent.train(nb_iterations=10, log_interval=1)
    print(rewards)

if __name__ == "__main__":
    test()
