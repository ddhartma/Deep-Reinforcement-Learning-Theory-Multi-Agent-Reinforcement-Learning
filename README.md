[image1]: assets/1.png 
[image2]: assets/2.png 
[image3]: assets/3.png 
[image4]: assets/4.png 
[image5]: assets/5.png 
[image6]: assets/6.png 
[image7]: assets/7.png 
[image8]: assets/8.png 
[image9]: assets/9.png 
[image10]: assets/10.png 
[image11]: assets/11.png 
[image12]: assets/12.png 
[image13]: assets/13.png 
[image14]: assets/14.png 
[image15]: assets/15.png 
[image16]: assets/16.png 
[image17]: assets/17.png 
[image18]: assets/18.png 


# Deep Reinforcement Learning Theory - Multi-Agent Reinforcement Learning

Let's introduce the concept of multi-agent RL, also known as MARL. 

## Content 
- [Introduction](#intro)
- [Motivation for Multi-agent systems](#motivation)
- [Markov Games](#markov_games)
- [Approaches to MARL](#approaches)
- [Paper: Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments](#paper)
- [Physical Deception problem: Code implementation](#coding)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="intro"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

### Overview:
- ***Policy-Based Methods***: methods such as 
    - hill climbing
    - simulated annealing
    - adaptive noise scaling
    - cross-entropy methods
    - evolution strategies

- ***Policy Gradient Methods***:
    - REINFORCE
    - lower the variance of policy gradient algorithms

- ***Proximal Policy Optimization***:
    - Proximal Policy Optimization (PPO), a cutting-edge policy gradient method

- ***Actor-Critic Methods***
    - how to combine value-based and policy-based methods
    - bringing together the best of both worlds, to solve challenging reinforcement learning problems

## Motivation for Multi-agent systems <a name="motivation"></a>
- We live in  a multi agent world
- We do not become intelligent in isolation
    - As a baby, interactions with our parents shape us  
    - In school, we learn to collaborate and compete with others. 
    - Intelligence is the result of interactions with multiple agents over our lifetime.

- Intelligent agents have to interact with humans and with other agents

- There are different kinds of interactions going on between agents:
    - coordination
    - competition
    - communication
    - prediction
    - negotiation
    - and so on

### Some Examples:
- A group of drones or robots whose aim is to pick up a package and drop it to the destination is a multi-agent system.
- In the stock market, each person who is trading can be considered as an agent and the profit maximization process can be modeled as a multi-agent problem.
- Interactive robots or humanoids that interact with humans and get some task done are multi-agent systems if we consider humans to be agents.
- Windmills in a wind farm can be thought of as multiple agents. The wind turbines (agents) have to figure out the optimal direction to face by themselves, and obtained maximum energy from the wind farm.

    ![image1]

### Benefits and Challenges of MARL

| Benefits     | Challenges     |
| :------------- | :------------- |
| Agents **can share their experiences** with one another making each other smarter.      | Extra **hardware and software capabilities** for communication needed. |
| A multi-agent system is **robust**. Agents can be replaced with a **copy** when they fail. | Substituting agent now has to do some **extra work**. |
| **Scalability**, insertion of new agents becomes easy. | If more agents are added to the system, the **system becomes more complex** than before. |

## Markov Games <a name="markov_games"></a>
- **Single agent**: 
    - A **drone** with the task of grabbing a package.
    - **Actions**: are going right, left, up, down, and grasping.
    - **Reward**: 50 for grasping the package and -1 otherwise.

- **Multi-agent RL**:
    - **More** than one agent.
    - There is a **joint set of actions**.

    ![image2]


### Markov game framework,
- A Markov game, is a tuple written as this:

    ![image3]

    - **n** is the number of agents.
    - **S** is the set of states of the environment.
    - **A<sub>i</sub>** is the set of actions of each agent **i**.
    - **A** is the joint action space.
    - **O<sub>i</sub>** is the set of observations of agent **i**
    - **R<sub>i</sub>** is the reward function of agent **i**, which returns a real value for acting in a particular state.
    - **&pi;<sub>i</sub>** is the policy of each agent **i**, that given its observations, returns a probability distribution over the actions **A<sub>i</sub>**.
    - **T** is the state transition function. Given the **current state** and the **joint action**,
    it provides a **probability distribution over the set of possible next states**. 
    
- State transitions are Markovian, just like in an MDP.
- Markovian means that the next state depends only on the present state and the actions taken in this state.


## Approaches to MARL <a name="approaches"></a> 
Adapting single-agent techniques to the multi-agent case

### Approach 1: Non-stationarity environment
- **Train all the agents independently** without considering the existence of other agents.
- Agent considers all the others to be a **part of the environment** and learns its **own policy**.
- Since all are learning simultaneously, the **environment** as seen from the prospective of a single agent, **changes dynamically**. --> Non-stationarity environment
- In most single agent algorithms the environment is stationary, which leads to certain convergence guarantees. Under non-stationarity conditions, **convergence guarantees no longer hold**.

    ![image4]

### Approach 2: Matter agent
- Agent considers the existence of multiple agents.
- A **single policy** is learned for all the agents.
- **Input**: Present **state of the environment**
- **Output** the action of each agent in the form of a **single joint action vector**.
- Agents receive a **global reward**.
- Consider: The **joint action space** increases **exponentially with the number of agents**. If the environment is partially observable or the agents can only see locally, each agent will have a different observation of the environment state. So this approach works well only in **fully observable environments**.

    ![image5]

### Cooperation vs. Competition
- Agents can interact with each other in multiple forms:
    - **cooperative**
    - **competitive**
    - or both: **mixed cooperative and competetive** environment

- There is a hyperparameter called **Team Spirit**
    - 0 --> Agents only care about their individual reward function 
    - 1 --> Agents completely care about the team's reward function


## Paper: Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments <a name="paper"></a> 
- [2017, Lowe et al. Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)
- Paper implements a **multi-agent version of DDPG**.
- DDPG is an **off policy actor-critic** algorithm that uses the concept of **target networks**.
    - **Input to actor**: of the action network  is the **current state**
    - **Output of actor**: output is a **real value or a vector** representing an **action** chosen from a continuous action space.
- **Open AI** has created a multi-agent environment called **multi-agent particle**.
    - It consists of particles that is agents and some landmarks.
    - There are many scenarios for this environment. 
- Here: **physical deception** 
    - Many agents **cooperate** to reach the target landmark  out of end landmarks.
    - There is an adversary which is also trying to reach the target landmark, but it doesn't know which out of the end landmarks is the target landmark.

    ![image6]

    - Agents are **rewarded** based on the **least distance of any of the agents** to the landmark,
    - and **penalized** based on the **distance between the adversary and the target landmark**.
    - Under this reward structure, the **agents cooperate to spread out** across all the landmarks, to **deceive the adversary**.

    ![image7]


- Some extra information is used to ease training, but that information is not used during the testing time.
- This framework is implemented using an actor-critic algorithm.
    - Each agent has an **actor and an critic network**.
    - During **training**, the critic for each agent uses extra information like **states observed and actions taken** by all the other agents.
    - Learning the critic for each agent allows us to use a **different reward structure for each agent**. Hence, the algorithm can be used in all, cooperative, competitive, and mixed scenarios.
    - Each **actor** has access to **only its agent's observation and actions**.
    - During **testing** time, **only the actors** are present
    
    ![image8]


## Physical Deception problem: Code implementation <a name="coding"></a> 
- Open the folder ```workspace_physical_deception``` and check out the README file there.
- Train agents to solve the Physical Deception problem.
- **Goal of the environment**
    - Blue dots are the "good agents".
    - Red dot is an "adversary". 
    - All of the agents' goals are to go near the green target. 
    - The blue agents know which one is green, but the Red agent is color-blind and (does not know which target is green/black!)
    - **The optimal solution** is for the **red agent to chase one of the blue agent**, and for the **blue agents to split up** and go toward each of the target.


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Deep-Reinforcement-Learning-Theory-Actor-Critic-Methods.git
```

- Change Directory
```
$ cd Deep-Reinforcement-Learning-Theory-Actor-Critic-Methods
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name drl_env
```

- Activate the installed environment via
```
$ conda activate drl_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Very good summary of DQN](https://medium.com/@nisheed/udacity-deep-reinforcement-learning-project-1-navigation-d16b43793af5)
* [An Introduction to Deep Reinforcement Learning](https://thomassimonini.medium.com/an-introduction-to-deep-reinforcement-learning-17a565999c0c)
* Helpful medium blog post on policies [Off-policy vs On-Policy vs Offline Reinforcement Learning Demystified!](https://kowshikchilamkurthy.medium.com/off-policy-vs-on-policy-vs-offline-reinforcement-learning-demystified-f7f87e275b48)
* [Understanding Baseline Techniques for REINFORCE](https://medium.com/@fork.tree.ai/understanding-baseline-techniques-for-reinforce-53a1e2279b57)
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Cheat Sheet](https://towardsdatascience.com/reinforcement-learning-cheat-sheet-2f9453df7651)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)

Important publications
* [2004 Y. Ng et al., Autonomoushelicopterflightviareinforcementlearning --> Inverse Reinforcement Learning](https://people.eecs.berkeley.edu/~jordan/papers/ng-etal03.pdf)
* [2004 Kohl et al., Policy Gradient Reinforcement Learning for FastQuadrupedal Locomotion --> Policy Gradient Methods](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf)
* [2013-2015, Mnih et al. Human-level control through deep reinforcementlearning --> DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [2014, Silver et al., Deterministic Policy Gradient Algorithms --> DPG](http://proceedings.mlr.press/v32/silver14.html)
* [2015, Lillicrap et al., Continuous control with deep reinforcement learning --> DDPG](https://arxiv.org/abs/1509.02971)
* [2015, Schulman et al, High-Dimensional Continuous Control Using Generalized Advantage Estimation --> GAE](https://arxiv.org/abs/1506.02438)
* [2016, Schulman et al., Benchmarking Deep Reinforcement Learning for Continuous Control --> TRPO and GAE](https://arxiv.org/abs/1604.06778)
* [2017, PPO](https://openai.com/blog/openai-baselines-ppo/)
* [2018, Bart-Maron et al., Distributed Distributional Deterministic Policy Gradients](https://openreview.net/forum?id=SyZipzbCb)
* [2013, Sergey et al., Guided Policy Search --> GPS](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)
* [2015, van Hasselt et al., Deep Reinforcement Learning with Double Q-learning --> DDQN](https://arxiv.org/abs/1509.06461)
* [1993, Truhn et al., Issues in Using Function Approximation for Reinforcement Learning](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)
* [2015, Schaul et al., Prioritized Experience Replay --> PER](https://arxiv.org/abs/1511.05952)
* [2015, Wang et al., Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [2016, Silver et al., Mastering the game of Go with deep neural networks and tree search](https://www.researchgate.net/publication/292074166_Mastering_the_game_of_Go_with_deep_neural_networks_and_tree_search)
* [2017, Hessel et al. Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
* [2016, Mnih et al., Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [2017, Bellemare et al., A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [2017, Fortunato et al., Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
* [2016, Wang et al., Sample Efficient Actor-Critic with Experience Replay --> ACER](https://arxiv.org/abs/1611.01224)
* [2017, Lowe et al. Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)
