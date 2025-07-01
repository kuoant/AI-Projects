#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:21:00 2024

"""


import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


nAgents  = 100
nDays    = 1000
nInterPD = 50 # let's say 50 interactions per day, 50'000 total
T = nDays * nInterPD + 1   

eps   = 0.01  # probability: agent a performas a random type change
delta = 0.5   # probability: a copies b's type

g     = 1    # scalar for chartists' predictions
nu    = 0.5  # scalar for fundamentalists' predictions

sigma  = 0.001 # stdev for noisy signal
p_fund = 1.0   # fundamental price


# initialize 
chartist = rd.rand(nAgents) < 0.5  # probability being a chartist

frac_chart = np.full(T, chartist.mean() )
price      = np.full(T, p_fund)


for t in range(2, T):
    
    
    # mean reverse process, random change
    if rd.rand() < eps:   
        a = rd.randint(nAgents)
        chartist[a] = not chartist[a]
    
    # positive feedback process, copying someone else
    if rd.rand() < delta:
        
        a, b = rd.permutation(nAgents)[:2]  # pick two different agents
        chartist[a] = chartist[b]  # a copies b's type
    
    frac_chart[t] = chartist.mean()
    
    # price process
    E_chart = (price[t-1] - price[t-2]) * g  # chartist, positive feedback, repeat shock from yesterday
    E_fund  = (p_fund - price[t-1]) * nu     # fundamentalists, negative feedback, contrarian
    
    w_Chart = frac_chart[t]
    noise   = rd.randn() * sigma 
    
    dPrice  = w_Chart * E_chart + (1-w_Chart) * E_fund + noise
    price[t] = price[t-1] + dPrice


t = range(0, T, nInterPD)   # ignore intraday changes
r = np.diff(price[t])      # returns

fig, axs = plt.subplots(2,1)
axs[0].plot(frac_chart[t])
axs[0].set_ylim([0,1])
axs[1].plot(r)


# ABM Conclusion: We can see volatility clustering.
# The big shocks happen, because chartists are the majority and contributing to risk.

# Careful
# What if fundamentalists know that majority are chartists.
# They will switch strategy, even though they know assets are over-/underpriced.





