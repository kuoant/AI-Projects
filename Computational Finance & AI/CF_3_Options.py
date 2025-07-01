#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:54:06 2024

@title: Option Pricing
"""

import numpy as np
import numpy.random as rd
import scipy.stats as scs
import matplotlib.pyplot as plt


#%% Black Scholes

def BlackScholes_call(S0, K, r, sigma, T):
    
    d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T)  / (sigma * np.sqrt(T))
    term1 = S0 * scs.norm.cdf(d1)

    d2 = d1 - sigma * np.sqrt(T)
    term2 = K * np.exp(-r*T) * scs.norm.cdf(d2)
    
    c0 = term1 - term2
    
    return c0

# Black Scholes: Deisgned such that in the long you should only earn the risk-free
# Designed for European Options, which cannot be exercised before expiration


#%% Price a European option with Monte Carlo

# need to model payoff average that is equivalent with risk-free rate

def optEuropean_MC(S0, K, r, sigma, T, n_exp=1_000_000):

    rMC = rd.randn(n_exp) * sigma * np.sqrt(T) + (r - sigma**2 / 2) * T   # we need to correct for the bias of skewed distribution, risk-neutral distribution
    ST = S0 * np.exp(rMC)  # simulation of stock prices at maturity (risk neutral), E[ST] = S0 * exp(r*T)
    
    innerValue = lambda ST, K: np.maximum(ST-K, 0)   # can also have different payoff-structures for inner value
    # innerValue = lambda ST, K: np.maximum(K-ST, 0) # change K-ST for puts
    
    cT = innerValue(ST, K)

    # This will give us the result from Monte-Carlo
    c0 = np.mean(cT) * np.exp(-r*T)  # c0 = discounted expected cT

    return c0


#%% Binomial Trees


def BT_option(S0, K, r, sigma, T, M=100, American=False):
    
    dt = T / M    # Length per subperiod
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r*dt) - d) / (u-d)  # we are allowed to earn the risk-free rate

    # create tree for stock prices
    S = np.full((M+1, M+1), np.nan)
    S[0,0] = S0
    
    for i in range(1, M+1):
        j = i
        S[i, :j] = S[i-1, :j] * d   # current period = previous period
        S[i, j] = S[i-1, j-1] * u   # constructing the up-movements

        
    # Create tree for option prices    
    innerValue = lambda S, K: np.maximum(S-K, 0)
    
    # we want to go back through the tree (in order to calculate for American & European Options)
    opt = np.full((M+1, M+1), np.nan)
    opt[M] = innerValue(S[M], K)

    for i in range(M, 0, -1):  # for i = M, M-1, M-2, ..., 2, 1
        
        j = np.arange(0, i)
        opt[i-1, j] = p * opt[i,j+1] + (1-p) * opt[i, j]  # discounted weighted average of the next two nodes, opt[i,j] = stayed on the same level
        opt[i-1, j] = opt[i-1, j] * np.exp(-r*dt) # discounted for the length of 1 time stamp
        
        if American: # makes no different, if we don't have dividends
            opt[i-1,j] = np.maximum(opt[i-1, j], innerValue(S[i-1,j], K))   
        
    c0 = opt[0,0]
    
    return c0
    

# the more steps in the grid we include, the better the approximation of Black Scholes


#%% experiments

S0 = 100
r =  0.01
sigma = 0.4
T = 1
K = 100

c0_BS = BlackScholes_call(S0, K, r, sigma, T)
print("Black Scholes:\t", c0_BS)

c0_BS = optEuropean_MC(S0, K, r, sigma, T, n_exp=10_0000_000)
print("Monte Carlo:\t", c0_BS)

c0_BT = BT_option(S0, K, r, sigma, T, M=10000, American=True)
print("Binomial Tree:\t", c0_BT)


S = np.linspace(1, 120, 120)
c = [BlackScholes_call(S0, K, r, sigma, T) for S0 in S]
plt.plot(S, c)








