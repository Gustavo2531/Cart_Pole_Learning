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
        

def mc_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.15, epsilon_decay=0.998):
    
    # Initialize the basic stuff
    returns_sum = defaultdict(float) 
    returns_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))


#    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    j = 0 # counter for the number of episodes
    
    ## Train
    while j<n_episodes:
        j += 1
        epsilon = epsilon_decay ** j
        policy = epsilon_greedy_policy(Q, (0.998**j), env.action_space.n)
        s = env.reset()
        d = False
        i = 0 # counter for the number of steps needed to finish episode
        
        episode = []
        
        if j % 1000 == 00 and j>0:
            print("Now playing episode: ", j)

        ########################################################
        ####        ALGORITHM STARTS HERE
        ########################################################
        
        # Generate samples from the episode
        while not d:
            i +=1
            mixed_policy = policy(s)
            a = np.random.choice(np.arange(len(mixed_policy)), p=mixed_policy)
            s1, r, d, _ = env.step(a)
            episode.append((s,a,r))
            s = s1
    
        # Update value functions after the sampling from the episode
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        
        for s,a in sa_in_episode:
            sa_pair = (s,a)
            # Find first visit to (s,a)
            first_visit_idx = next(i for i,x in enumerate(episode) 
                                    if x[0] == s and x[1] == a )
            G = sum(x[2]*(gamma**i) for i, x in enumerate(episode[first_visit_idx:]))
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[s][a] = returns_sum[sa_pair]/returns_count[sa_pair]   

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
    env = gym.wrappers.Monitor(env, 'mc_glie', force = True)
    Q, policy = mc_epsilon_greedy(env,n_episodes=10000)
    V = create_value_function(Q)
    plot_value_function(V)
    env.close()  
    gym.upload('mc_glie', api_key=API_KEY)

    

   
