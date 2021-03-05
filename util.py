"""define tools"""

import base64
import IPython
import imageio
import numpy as np
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment


def embed_mp4(filename):
    """encode and show video in IPython"""
    video = open(filename, "rb").read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
      <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())
    return IPython.display.HTML(tag)


def create_atari_video(filename, atari, policy, nb_episodes=5, fps=30):
    """make videos for Atari games
    
    atari must be a OpenAI Gym atari env
    policy must have a action() method
    """
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(nb_episodes):
            done = False
            frame = atari.reset() # returns (210, 160, 3) array 
            while not done:
                # add a new frame
                video.append_data(frame)
                # interact with atari
                action = policy.action(frame)
                frame, _, done, _ = atari.step(action)
    return embed_mp4(filename)


def create_tf_env_video(filename, env, policy, nb_episodes=5, fps=30):
    """make videos for TFPyEnvironment games."""
    assert(isinstance(env, TFPyEnvironment))

    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(nb_episodes):
            time_step = env.reset()
            frame = np.squeeze(env.pyenv.render()) # returns (210, 160, 3) array
            while not time_step.is_last():
                # add a new frame
                video.append_data(frame)
                # interact with games
                action = policy.action(time_step)
                time_step = env.step(action)
                frame = np.squeeze(env.pyenv.render())
    return embed_mp4(filename)


def load_policy(dirname):
    return tf.compat.v2.saved_model.load(dirname)

