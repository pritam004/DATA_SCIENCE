#!/usr/bin/env python
# coding: utf-8

# Install OpenAI Gym 4 . Implement a neural network based policy-gradient solution for the
# CartPole-v1 and MountainCarContinuous-v0 environments from OpenAI Gym. Plot episode rewards as a
# function of the number of training episodes and save it as “2.png”. As in the case of Problem 1, you are free
# to choose the architecture of the policy neural networks. Specify all the hyper-parameters used by you along
# with the training strategy

# In[ ]:


'''All codes and examples are reproduced as given in the book 
Deep Reinforcement Learning Hands-On by Maxim Lapan'''


# In[4]:


#!/usr/bin/env python3
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# In the beginning, we define hyperparameters (imports are omitted). The
# EPISODES_TO_TRAIN value specifies how many complete episodes we’ll use for
# training.
# 
# Entropy beta
# value is the scale of the entropy bonus. The REWARD_STEPS value specifies how
# many steps ahead the Bellman equation is unrolled to estimate the discounted
# total reward of every transition.

# In[5]:


GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8

REWARD_STEPS = 10


# Note that despite the fact our
# network returns probabilities, we’re not applying softmax nonlinearity to the
# output. The reason behind this is that we’ll use the PyTorch log_softmax
# function to calculate the logarithm of the softmax output at once. This way of
# calculation is much more numerically stable, but we need to remember that
# output from the network is not probability, but raw scores (usually called logits).

# In[6]:


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# In[7]:





if __name__ == "__main__":
    
    rew=[]
    ep=[]
    
    env = gym.make("CartPole-v1")
    env = gym.wrappers.Monitor(env, "recording",force=True)
    writer = SummaryWriter(comment="-cartpole-pg")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
    
    
    
    #source is asked to unroll the Bellman equation for 10 steps
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0

    batch_states, batch_actions, batch_scales = [], [], []
    
    '''In the training loop, we maintain the sum of the discounted reward for every
transition and use it to calculate the baseline for policy scale.'''

    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            ep.append(done_episodes)
            rew.append(mean_rewards)
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)
        
        '''Then we add the entropy bonus to the loss by calculating the entropy of the
batch and subtracting it from the loss. As entropy has a maximum for uniform
probability distribution and we want to push the training towards this maximum,
we need to subtract from the loss'''
        
        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        optimizer.step()

        # calc KL-div
        '''Then, we calculate the Kullback-Leibler (KL)-divergence between the new and
the old policy. KL-divergence is an information theory measurement of how one
probability distribution diverges from another expected probability distribution.
In our example, it is being used to compare the policy returned by the model
before and after the optimization step. High spikes in KL are usually a bad sign,
showing that our policy was pushed too far from the previous policy, which is a
bad idea most of the time (as our NN is a very nonlinear function in a high-dimension space, so large changes in the model weight could have a very strong
influence on policy)'''
        
        
        
        
        
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)
        
        
        
        '''Finally, we calculate the statistics about the gradients on this training step. It’s
usually good practice to show the graph of maximum and L2-norm of gradients
to get an idea about the training dynamics.'''

        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

            
        '''summary dropped in tensorboard'''
            
            
            
            
        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy_v.item(), step_idx)
        writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)
        writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        writer.add_scalar("grad_max", grad_max, step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()
    env.close()
    env.env.close()
    writer.close()


# In[ ]:





# In[ ]:





# In[ ]:




