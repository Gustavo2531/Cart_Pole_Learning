# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:32:34 2017

@author: pc
"""

API_KEY = 'sk_0oS8pMQ3TbmomIWSD3UuQ'


## EXERCISES: Monte-Carlo methods
import gym
import numpy as np
from collections import defaultdict
import seaborn as sns
import itertools
import sys


def epsilon_greedy_policy(Q, epsilon, n_actions):
    """
    Creates an epsilon-greedy policy based on the given Q-function and epsilon.
    Args:
        Q: A dict that maps (s,a) to float.
        epsilon: float between 0 and 1, the probability of choosing a sub-optimal action.
        n_actions: the total actions available.
    """
    
    def policy_fn(state):
        prob = np.ones(n_actions, dtype = float)*epsilon/n_actions
        best_action = np.argmax(Q[state])
        prob[best_action] += (1.0-epsilon)
        return prob
    return policy_fn



def qLearning(env, n_episodes, discount_factor=0.9998, alpha=0.1, epsilon=0.1):
    
    # Initialize the basic stuff
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    j = 0
    
    
    while j<n_episodes:
        j += 1
        state = env.reset()
        d = False
        i = 0
        episode = []
        
        if j % 1000 == 00 and j>0:
            print("Now playing episode: ", j)
    
        ########################################################
        ####        ALGORITHM STARTS HERE
        ########################################################
        
        # Generate samples from the episode
        while not d:
            i +=1
            policy = epsilon_greedy_policy(Q, 1-(j/n_episodes), env.action_space.n)
            mixed_policy = policy(state)
            a = np.random.choice(np.arange(len(mixed_policy)), p=mixed_policy)
            next_state, r, d, _ = env.step(a)
            episode.append((state,a,r))
            
            Q[state][a] = Q[state][a] + 0.1 * (r + 0.991*np.max(Q[next_state][a])-Q[state][a])
            state = next_state
        
        # Update value functions after the sampling from the episode
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        
        for state,a in sa_in_episode:
            sa_pair = (state,a)
            
            # Find first visit to (s,a)
            first_visit_idx = next(i for i,x in enumerate(episode)
                                   if x[0] == state and x[1] == a )
            G = sum(x[2]*(discount_factor**i) for i, x in enumerate(episode[first_visit_idx:]))
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][a] = returns_sum[sa_pair]/returns_count[sa_pair]
            

########################################################
####        ALGORITHM ENDS HERE
########################################################

    return Q, policy




def create_value_function(Q):
    V = defaultdict(float)
    for s, actions in Q.items():
        action_val = np.max(actions)        
        V[s] = action_val
    return V


def plot_value_function(V):
    A = np.ones(16)*(-1)
    for k in V.keys():
        A[k] = V[k]
    sns.heatmap(A.reshape((4,4)))
        

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    env = gym.wrappers.Monitor(env, 'qlearning', force = True)
    Q, policy = qLearning(env,n_episodes=50000)
    V = create_value_function(Q)
    plot_value_function(V)
    env.close()  
    gym.upload('qlearning', api_key=API_KEY)

    

   
