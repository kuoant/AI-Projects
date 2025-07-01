#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:20:34 2024

"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import CFAI_support as cfai

#%% Testbed + Strategies


def chartist(S, V0=1, rule = lambda S_t: 1):
    
    # rule(S) returns the "optimal" weight for the risky asset
    
    pf = cfai.portfolio(S0 = S[0], V0=V0)     # create test portfolio
    
    T = S.shape[0]
    
    path = [pf.value]

    for t in range(1, T):
        
        # at beginning of the period
        
        weight = rule(S[:t])   # find target weight based prices S_i with i = 0,1,....,(t-1) up to yesterday
        
        pf.rebalance(weight)
        
        # end of period
        
        pf.updatePrice(S[t])   # set price of risky asset to S_t and pay interest on cash
        
        path.append(pf.value)

    return path


#%% Rules


def rule_BaH(S):
    wOpt = 1
    return wOpt

# check ln(S_t/S_t-1) > 0, can also directly use S_t > S_t-1, easier
def rule_TrendFollower(S):
    
    if len(S) < 2:
        wOpt = 1
    
    elif S[-1] >= S[-2]:   # if price went up
        wOpt = 1
    
    else:
        wOpt = 0
        
    return wOpt


def rule_Contrarian(S):
    
    if len(S) < 2:
        wOpt = 1
    
    elif S[-1] <= S[-2]:   # if price went down, hold; otherwise cash only
        wOpt = 1
    
    else:
        wOpt = 0
        
    return wOpt
    


#%% Experiments

rM, rA = cfai.readReturns(start="2015", before="2017", indAssets=[1])  # get returns considered asset
E_r, Cov, beta, rA_filter = cfai.CAPM_estimator(rM, rA)  # ... and the corresponding parameters & filtered returns


S_sim = cfai.simulatePaths("GBM", E_r=E_r, Cov=Cov, rEmp=rA)

Rules = [rule_BaH, rule_TrendFollower, rule_Contrarian]


path = [chartist(S_sim, rule = R) for R in Rules]
path = np.array(path).T

# plt.plot(S_sim, "k")
plt.plot(path)
plt.legend([R.__name__ for R in Rules])

# all rules perform equally well, because there is no momentum in GBM

#%% B&H vs. Trendfollower vs. ruleContrarian

nExp = 1000
Rules = [rule_BaH, rule_TrendFollower, rule_Contrarian]
VT = np.ones((nExp, len(Rules)))

for e in range(nExp):
    
    S_sim = cfai.simulatePaths("boot", rEmp=rA_filter, b=10)
    VT[e] = [chartist(S_sim, rule=R)[-1] for R in Rules]    

# B&H spreads most because we are always invested, not surprising
# Backtesting/Bootstrapping here more useful than theoretic GBM, because we have no momentum     
plt.boxplot(VT)
VT.mean(0)

# rA_filter = rA - rA.mean() + E_r*dt


    
    
    