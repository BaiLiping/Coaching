This is the repository for paper:
Accelerate Reinforcement Training with Prohibitive Boundary

Install the Required Environment and Agent:
```
pip3 install gym
pip3 install tensorforce
```

There are various implementations of reinforcement learning agents with either PyTorch or Tensorforce. In this paper, we choose Tensorforce, which is a set of implementations maintained by a team in Cambridge. The details of this repository can be found at: https://github.com/tensorforce/tensorforce. 

We use Mujoco environment in our experiments as well. If you want to run the code, you need to obtain the Mujoco License first. 
One of the issues when it comes to Mujoco is the difficulty to sort out its observation space. To find the detailed logs of how to query into Mujoco environment please refer to https://bailiping.github.io/Mujoco/

If you want to validate the code, all you have to do is go to the folder of the environment you are interested in, and run the code. Please note that the purpose of those code is to validate the hypothesis that prohibitive boundaries accelerate reinforcement learning training. So the parameters are tuned not to minimize training time but to acceturate the difference between agents trained with prohibitive boundaries and the ones without.

When you run the code, the trained agents would also be evaluated on an environment without any prohibitive boundary.
