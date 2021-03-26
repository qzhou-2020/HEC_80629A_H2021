from tfdqn import config
from tfdqn import TFDqnAgent


# env
config["env"]["name"] = "CartPole-v0"

# agent
config["agent"]["type"] = "ddqn"
config["agent"]["structure"] = [
    (32, {"activation": "relu"}),
    (32, {"activation": "relu"}),
]
config["agent"]["optimizer"]["learning_rate"] = 5e-4
config["agent"]["gamma"]=1.0
config["agent"]["target"]["soft"] = 1e-3
config["agent"]["target"]["period"] = 4

# buffer
config["replay_buffer"]["capacity"] = 10000
config["replay_buffer"]["batch_size"] = 64

# driver
config["driver"]["length"] = 1

# saver
config["policy_saver"] = False

dqn = TFDqnAgent(config)
dqn.train(nb_iterations=2000, log_interval=50)