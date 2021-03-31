# configuration file

```python
{
    "env": {
        "name": "LunarLander-v2",           # env name
        "is_atari": false,                  # true if it's an Atari
        "use_wrapper_if_atari": true,       # true if use wrapper
        "wrapper": {
            "noop_max": 30,                 # max no-op at beginning
            "frame_skip": 4,                # see only 1 of number of frames
            "screen_size": 84,              # resize the output
            "terminal_on_life_loss": true,  # episode ends on life loss
            "clip_reward": true             # true, reward = -/+1
        }
    },
    "agent": {
        "type": "ddqn",                     # "dqn" or "ddqn"
        "network": {
            "type": "dense",                # "dense", "conv", or "dense-duel"
            "hidden_layer_size": [64, 64],  # layer size, here there are two hidden layers
            "state_hidden_layer": [256],    # if dense-duel, size of state layer
            "advance_hidden_layer": [256]   # if dense-duel, size of advance layer
        },
        "explore": {
            "type": "decay_epsilon",        # "epsilon" or "decay_epsilon"
            "epsilon": 0.1,                 # constant eps if "epsilon", min eps if "decay_epsilon"
            "decay_rate": 1e-5              # decay speed if "decay_epsilon"
        },
        "optimizer": {      
            "type": "adam",                 # optimizer
            "loss": "mse",                  # loss function, "mse", "mae", or "huber"
            "learning_rate": 2.5e-4         # learning rate
        },
        "gamma": 0.99                       # discounting factor
    },
    "buffer": {
        "use_per": false,                   # true if use Prioritized-Experience-Replay buffer
        "size": 100000,                     # max buffer size, buffer is circular
        "alpha": 0.6,                       # hyper-parameter for PER
        "beta": 0.4,                        # hyper-parameter for PER
        "anneal_alpha_rate": 1e-5,          # hyper-parameter for PER
        "anneal_beta_rate": 1e-5            # hyper-parameter for PER
    },
    "train": {
        "mode": "episode",                  # "episode" or "step"
        "length": 2000,                     # length of "episode" or "step"
        "batch_size": 32,                   # batch size for learning
        "max_steps_per_episode": 1000,      # terminate when the steps in an episode exceeds this number
        "warm_up": 1000,                    # skip learning for the first steps
        "online_update_period": 1,          # update every 1 step(s)
        "target_sync_period": 1000,         # sync target net every number of steps
        "use_soft_target_update": false,    # use soft target net update
        "target_soft_update_tau": 0.1,      # update factor if use soft update
        "early_stop": false,                # true if use early stop. (not implemented)
        "log_interval": 50,                 # log every number of steps
        "display_average_reward": true,     # true if show average episodic reward in log
        "display_max_reward": true          # true if show max episodic reward in log
    },
    "misc": {
        "save_final_model": true,           # true if save final model
        "save_intermediate_model_at": [],   # save model at certain "episode"
        "save_model_to": "/content/drive/MyDrive/RL/study/case_8",  # save model to
        "save_log": true,                                           # true if save log to a file
        "save_log_to": "/content/drive/MyDrive/RL/study/case_8.log" # log file location
    }
}
```

# empirical study plan

We are testing several hyperparameters in this study.

| case index | type of agent |    exploration policy   | type of experience replay | replay buffer size | batch size | learn frequency | synchronization frequency | gamma |
|:----------:|:-------------:|:-----------------------:|:-------------------------:|:------------------:|:----------:|:---------------:|:-------------------------:|:-----:|
|      1     |      DQN      | constant epsilon        |           normal          |      $2^{20}$      |     32     |        1        |           10              |  0.99 |
|      2     |      DQN      | constant epsilon        |           normal          |      $2^{20}$      |     32     |        1        |           1000            |  0.99 |
|      3     |      DQN      |   decay epsilon         |           normal          |      $2^{20}$      |     32     |        1        |           10              |  0.99 |
|      4     |      DQN      |   decay epsilon         |           normal          |      $2^{20}$      |     32     |        1        |           1000            |  0.99 |
|      5     |      DQN      |   decay epsilon         |           normal          |      $2^{17}$      |     32     |        1        |           1000            |  0.99 |
|      6     |      DQN      |   decay epsilon         |           PER             |      $2^{20}$      |     32     |        1        |           1000            |  0.99 |
|      7     |      DQN      |   decay epsilon         |          decay PER        |      $2^{20}$      |     32     |        1        |           1000            |  0.99 |
|      8     |     DDQN      |   decay epsilon         |           normal          |      $2^{17}$      |     32     |        1        |           1000            |  0.99 |
|      9     |     DDQN      |   decay epsilon         |          decay PER        |      $2^{17}$      |     32     |        1        |           1000            |  0.99 |
|      10    |     DDQN      |   decay epsilon         |          decay PER        |      $2^{20}$      |     32     |        1        |           1000            |  0.99 |
|      11    |     D3QN      |   decay epsilon         |           normal          |      $2^{17}$      |     32     |        1        |           1000            |  0.99 |
|      12    |     D3QN      |   decay epsilon         |          decay PER        |      $2^{17}$      |     32     |        1        |           1000            |  0.99 |
|      5-1   |      DQN      |   decay epsilon         |           normal          |      $2^{17}$      |     32     |        2        |           1000            |  0.99 |
|      5-2   |      DQN      |   decay epsilon         |           normal          |      $2^{17}$      |     128    |        1        |           1000            |  0.99 |
|      6-1   |      DQN      |   decay epsilon         |           PER             |      $2^{17}$      |     32     |        1        |           1000            |  0.99 |


footnotes:

- $2^{17} = 131,702$, $2^{18} = 262,144$, $2^{19} = 524,288$, $2^{20} = 1,048,576$

- Case 1 and 2 fail to converge, so does case 4.

- LunarLander-v2, 2000 episodes, about ~500,000 steps