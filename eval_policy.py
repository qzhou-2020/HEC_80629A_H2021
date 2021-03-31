import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create a virtual display for running on a server
import pyvirtualdisplay
display = pyvirtualdisplay.Display(visible=False, size=(1400,900)).start()

import argparse
import base64
import imageio
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import gym

from agent import Policy
from wrapper import AtariWrapper


def eval(p, env, n, save_frame=False) -> list:
    history = []
    frames = []
    for _ in range(n):
        state = env.reset()
        done = False
        rewards = 0
        while not done:
            if save_frame:
                frames.append(env.render("rgb_array"))
            action = p.choose_action(state)
            state, reward, done, _ = env.step(action)
            rewards += reward
        history.append(rewards)
    return history, frames


def statistics(history):
    _max = max(history)
    _min = min(history)
    _mean = np.mean(history)
    _std = np.std(history)
    logging.info(f"episodic reward statistics: mean={_mean:.2f}, stdev={_std:.2f}")


def save_to_gif(p, env, filename, dpi, fps, repeat, interval=50):
    _, frames = eval(p, env, repeat, save_frame=True)
    plt.figure(figsize=(frames[0].shape[1]/dpi, frames[0].shape[0]/dpi), dpi=dpi)
    patch = plt.imshow(frames[0])
    plt.axis("off")
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)
    anim.save(filename, writer="imagemagick", fps=fps)


def save_to_mp4(p, env, filename, fps, repeat):
    _, frames = eval(p, env, repeat, save_frame=True)
    with imageio.get_writer(filename, fps=fps) as video:
        for frame in frames:
            video.append_data(frame)


if __name__ == "__main__":
    # handle cli
    parser = argparse.ArgumentParser(description="evaluate a policy")
    parser.add_argument("policy_dir", type=str)
    parser.add_argument("env_name", type=str)
    parser.add_argument("--atari", action="store_true")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-to", type=str, default="example.gif", help="save as gif or mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=72)
    parser.add_argument("--repeat", type=int, default=3)

    args = parser.parse_args()

    # load policy
    p = Policy(args.policy_dir)

    # load env
    env = gym.make(args.env_name)
    if args.atari:
        env = AtariWrapper(env)

    # evalulate
    history, _ = eval(p, env, args.runs)
    statistics(history)

    if not args.save:
        exit(0)

    suffix = args.save_to.split(".")[-1]
    if suffix == "gif":
        save_to_gif(p, env, args.save_to, args.dpi, args.fps, args.repeat)
    elif suffix == "mp4": 
        save_to_mp4(p, env, args.save_to, args.fps, args.repeat)
    else:
        logger.error(f"output format '{suffix}' is not supported")
