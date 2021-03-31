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
|      6-1   |      DQN      |   decay epsilon         |           PER             |      $2^{17}$      |     32     |        1        |           1000            |  0.99 |




footnotes:

- $2^{17} = 131,702$, $2^{18} = 262,144$, $2^{19} = 524,288$, $2^{20} = 1,048,576$

- Case 1 and 2 fail to converge. use Case 3 as baseline  

- LunarLander-v2, 2000 episodes, about ~500,000 steps
