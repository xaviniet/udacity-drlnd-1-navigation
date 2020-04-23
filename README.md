[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent! 


We have 3 files:
- [model.py](model.py) - the definition of the neural network for the Q-Learning 
- [nav_agent.py](nav_agent.py) (and [nav_agent_ddqn.py](nav_agent_ddqn.py)) - the agent that interacts with the environment
- [Navigation.ipynb](Navigation.ipynb) - Here we make the experiments and train the agent(s).

#### Navigation.ipynb

The notebook where we train the agent(s). 

- We import the needed modules


- We initialize the environment
```python 
env = UnityEnvironment(file_name="Banana.app")
```  


- We initialize the agent
```python
dqn_agent = Nav_agent(state_size= state_size, 
                      action_size=brain.vector_action_space_size, seed=42)
```  


- We define the function `dqn_nav` for training the agent  
```python
def dqn_nav(agent, env, n_episodes=2000, eps_start=1.0, eps_end=0.001, eps_decay=0.995, t_mode=False, solved=13.0, fcheckpoint='checkpoint.pth'):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        t_mode (boolean): True for train the agent and the mode of environment
        solved (float): The value of the average for solving the environment. Default: 13.0
        fcheckpoint (string): The file where we want to save the neural network weights
    """
 ```


- We train the agent calling the function and observe the results.

#### model.py
In this file we define the QNetwork class, and define the structure. 

#### nav_agent.py (and nav_agent_ddqn.py)

We define the Nav_agent class and methods for interacting (`act`,`step`) and training (`learn`, `soft_update`).
Also, we define the ReplayBuffer class, needed by the agent on the training of the algorithms coded.



