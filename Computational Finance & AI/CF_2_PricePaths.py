#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:23:32 2024

@title: Price Paths
"""

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


#%% Simulate prices with Geometric Brownian Motion at end of Period, Exercise 1.13


S0 = 50             # initial price
mu = 0.1            # drift p.a.
sigma = 0.25        # volatility p.a.

T = 1               # entire horizon in years
dt = 1 / 250        # distance between two observations (in years), e.g. 1/12 would be monthly
spy = 1/dt          # subperiods or steps per year?
M = int(T * spy)    # total number of subperiods

n_exp = 500                          # rows as different point in times

rt = rd.randn(M+1, n_exp) * sigma * np.sqrt(dt) + mu * dt # simulate returns
rt[0] = 0                              # set first return at 0
St = S0 * np.exp(rt.cumsum(0))         # calculate end-of-year prices

plt.plot(St)
    
# Probability that end-of-horizon price is below 40
prob_below = np.mean(St[-1] < 40)   
print(prob_below)

# Check all who fell below the threshold during the whole period
prob_below_once = np.mean(np.any(St < 40, axis=0))
print(prob_below_once)

# We can see that in the long-run the drift kicks in (lower probability of 
# making a substantial loss), but inbetween it can happen


# What is the expected price after 1 year?
E_ST_theo = S0 * np.exp(mu + sigma**2 / 2)
print(E_ST_theo)
print(St[250].mean()) 
# Warning: we are skewed, as long as log-returns are symmetric











