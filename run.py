import argparse
parser = argparse.ArgumentParser("train a RL agent")
parser.add_argument("--conf", type=str, default="./default.json", help="specify a json file as configuration")
args = parser.parse_args()

import json
config = json.load(open(args.conf, "r"))

import logging
import sys
LOGLEVEL = logging.INFO
if config["misc"]["save_log"]:
    logging.basicConfig(level=LOGLEVEL, filename=config["misc"]["save_log_to"])
else:
    logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)

# main body

import gym
from agent import DQN, train
from wrapper import AtariWrapper
from replay import UniformReplayBuffer, PrioritizedReplayBuffer
from observer import AverageObserver, MaximumObserver

env = gym.make(config["env"]["name"])
if config["env"]["is_atari"]:
    env = AtariWrapper(env, **config["env"]["wrapper"])

agent = DQN(
    config["agent"],
    env.observation_space.shape,
    env.action_space.n,
)

if config["buffer"]["use_per"]:
    buffer = PrioritizedReplayBuffer(
        size = config["buffer"]["size"],
        alpha = config["buffer"]["alpha"],
        beta = config["buffer"]["beta"],
        anneal_alpha_rate = config["buffer"]["anneal_alpha_rate"],
        anneal_beta_rate = config["buffer"]["anneal_beta_rate"]
    )
else:
    buffer = UniformReplayBuffer(config["buffer"]["size"])

observer = []
if config["train"]["display_average_reward"]:
    observer.append(AverageObserver(config["train"]["log_interval"]))
if config["train"]["display_max_reward"]:
    observer.append(MaximumObserver(config["train"]["log_interval"]))

c = config["train"]
c.update(config["misc"])
c["observer"] = observer
history = train(env, agent, buffer, **c)
logging.info(history)
