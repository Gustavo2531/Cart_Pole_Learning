# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:51:13 2017

@author: pc
"""

import gym 
import numpy as np


env = gym.make('CartPole-v0')

best_weights = np.random.rand(4) * 2 - 1
best_reward = 0

# training loop
for i in range(10000):
    
    s = env.reset()
    total_reward = 0
    
    # episode loop
    for j in range(200):
        env.render()
        #env.step(env.action_space.sample())
        ###########################################
        # Assign the weights by an update rule
        ###########################################

#        random_noise = np.random.rand(4)
        weights = np.random.rand(4) * 2 - 1
#best_weights + 0.3 * random_noise

        # Choose action using the weights
        a = 1 if np.matmul(weights, s)<0 else 0
        
        
        
        s1, r, d, _ = env.step(a)    
        total_reward +=r
        
        if d:
            print(best_reward)
            break
    if (total_reward >= best_reward):
        best_reward = total_reward
        best_weights = weights
### YOUR TASK:
### - Change the update rule to random search (5 points)
### - EXTRA CREDIT (5p): use hill-climbing or a genetic algorithm to update the weights    
### - Implement some checks to see if the current weights are better or not            
###
print(best_reward)
env.close()        
