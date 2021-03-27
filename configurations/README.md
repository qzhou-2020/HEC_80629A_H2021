We are testing several hyperparameters in this study.

| case index | type of agent |    exploration policy   | type of experience replay | replay buffer size | batch size | learn frequency | synchronization frequency | gamma |
|:----------:|:-------------:|:-----------------------:|:-------------------------:|:------------------:|:----------:|:---------------:|:-------------------------:|:-----:|
|      1     |      DQN      | constant epsilon        |           normal          |      $2^{20}$      |     32     |        1        |           10              |  0.99 |
|      2     |      DQN      | annealing epsilon       |           normal          |      $2^{20}$      |     32     |        1        |           10              |  0.99 |
|      3     |      DQN      | annealing epsilon       |           normal          |      $2^{20}$      |     32     |        10       |           10              |  0.99 |
|      4     |      DQN      | annealing epsilon       |           normal          |      $2^{20}$      |     32     |        1        |           1000            |  0.99 |
|      5     |      DQN      | annealing epsilon       |           normal          |      $2^{17}$      |     32     |        1        |           1000            |  0.99 |
|      6     |      DQN      | annealing epsilon       |           PER             |      $2^{17}$      |     32     |        1        |           1000            |  0.99 |
|      7     |      DQN      | annealing epsilon       |           PER             |      $2^{20}$      |     32     |        1        |           1000            |  0.99 |
|      8     |      DQN      | annealing epsilon       |           PER             |      $2^{20}$      |     128    |        1        |           1000            |  0.99 |
|      9     |      DQN      | annealing epsilon       |        annealing PER      |      $2^{20}$      |     32     |        1        |           1000            |  0.99 |




footnotes:

- $2^{17} = 131,702$, $2^{18} = 262,144$, $2^{19} = 524,288$, $2^{20} = 1,048,576$

- Case 1 is the baseline case. 

- LunarLander-v2, 2000 episodes, about ~500,000 steps
