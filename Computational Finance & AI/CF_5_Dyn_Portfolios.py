#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:49:15 2024

"""

import numpy as np
import matplotlib.pyplot as plt
import CFAI_support as cfai


#%% CFAI


# Get market and asset returns
assets = [1,2,3]
N = len(assets)
rM, rA = cfai.readReturns(start="2015", before="2018-07-01", indAssets=assets)

# *rest collects the other output of the function
E_r, Cov, *rest = cfai.CAPM_estimator(rM, rA)

# Ssimulate by GBM
S_sim = cfai.simulatePaths("GBM", E_r=E_r, Cov=Cov)
plt.plot(S_sim)
plt.show()

# Simulate by bootstrapping
S_sim = cfai.simulatePaths("boot", E_r=E_r, Cov=Cov, rEmp=rA, b=10)
plt.plot(S_sim)

# Construct a portfolio
pf = cfai.portfolio(S0 = [10], V0 = 10000)
print(pf)



#%% Buy & Hold Strategy


def BuyAndHold(S_sim, wOpt):
    
    pf = cfai.portfolio(S_sim[0])
    pf.rebalance(wOpt)

    path = [pf.value]

    for t in range(1, S_sim.shape[0]):
    
        pf.updatePrice(S_sim[t])
    
        path.append(pf.value)  # what's the latest portfolio value

    return np.array(path)


#%% Rebalancing Strategy


# Return to wOpt in each period and consider "Stop Loss" and "Take Profit"
def Rebalance(S_sim, wOpt, lossCrit=-1, profitCrit=10000):
    
    pf = cfai.portfolio(S_sim[0])
    
    path = [pf.value]

    for t in range(1, S_sim.shape[0]):
        
        PaL_to_date = pf.value / pf.V0 - 1 
        
        if lossCrit < PaL_to_date < profitCrit:        
            pf.rebalance(wOpt)
        else:
            pf.rebalance(0)
    
        pf.updatePrice(S_sim[t])
    
        path.append(pf.value)  # what's the latest portfolio value

    return np.array(path)



#%% Utility Function


def util(V, g):
    
    if g==1:
        return np.log(V)
    
    else:
        return (V**(1-g) - 1) / (1-g)
    

V = np.linspace(0.8, 1.5, 51)  # different final values
Gamma = [0,1,3,5,7,10]         # different risk aversions

for g in Gamma:
    plt.plot(V, util(V,g))  # just for visualation

plt.legend(Gamma)
    
# The higher the risk-aversion, the more painful the loss is
# Measure for each scenario, how happy the investor is
# Based on that we can determine expected value of happiness




#%% B&H vs. Rebalancing - Single Run


S_sim = cfai.simulatePaths("boot", E_r=E_r, Cov=Cov, rEmp=rA, b=10)
wEW = np.ones(N) / N  # equally weighted portfolio

path = BuyAndHold(S_sim, wEW)
pathSL = Rebalance(S_sim, wEW, lossCrit=-0.15, profitCrit=.20)

plt.plot(S_sim, "k")
plt.plot(path, "r")
plt.plot(pathSL, "g")


#%% B&H vs. Rebalancing - Multiple Runs


nExp = 1000 

VT   = np.zeros((nExp, 2)) 
wOpt = np.ones(N) / N


# This allows to analyse the two strategies
for e in range(nExp):
    
    # Bootstrap = sort of Backtesting
    S_sim = cfai.simulatePaths("boot", E_r=E_r, Cov=Cov, rEmp=rA, b=10)
    
    pathBaH = BuyAndHold(S_sim, wOpt) 
    pathReb = Rebalance(S_sim, wOpt)    
    
    VT[e] = [pathBaH[-1], pathReb[-1]]


# Must assume that simulation is realistic scenario
# We keep some momentum if we block bootstrap

plt.boxplot(VT)
plt.show()

plt.hist(VT)
plt.show()

print(np.mean(VT < 1.0, axis=0))
print(np.mean(VT, axis=0))


# Let's test: H0 diff = 0, H1 diff!=0
diff = VT[:, 0] - VT[:, 1]
t = diff.mean() / diff.std()
print(t)  # not significant


# Advantages // Rebalancing: Nicer distribution // B&H: Keep the momentum

# Often Rebalancing is a bit worse, but not sigificant as we saw in testing.
# If we choose GBM instead of Bootstrapping, we don't have momentum anymore.
    

#%% Utility experiment with Stop Loss Rebalancing Strategy


nExp = 1000
SL = [-1, -.10, -.05] # different stop losses

VT = np.zeros((nExp, len(SL)))
wOpt = np.ones(N) / N

for e in range(nExp):
    
    S_sim = cfai.simulatePaths("boot", E_r=E_r, Cov=Cov, rEmp=rA, b=10)
    
    VT[e] = [Rebalance(S_sim, wOpt, lossCrit=s)[-1] for s in SL]



Gamma = [0,1,3,5,7,10,20,30] # different risk aversions

for g in Gamma:
    
    # columns = different SL, rows = different paths
    EU = util(VT, g).mean(axis=0) 
    print(f"gamma= {g:3} => E(U) = {EU}")
    

# Outcome with SL typically worse, no benefit from leaving the market early.
# But if we are highly risk-averse, loosing a dollar is really painful.

    
    
#%% CPPI


def CPPI(S_sim, V0, G, r, m=3):
    
    T = len(S_sim)
    
    V = [V0]
    F = []
    C = []
    E = []
    B = []
    
    for t in range(0,T):
    
        F.append(G * np.exp(-r * (T-t)))

        C.append(V[t] - F[t])
        
        C[t] = np.maximum(C[t],0)
        
        E.append(np.minimum(m * C[t], V[t]))
        
        B.append(V[t] - E[t])

        V.append( B[t] * np.exp(r) + E[t] * np.exp(S_sim[t]))
        
    return V


S_sim = [-1, -0.1, -0.1]
V0 = 200
Gt = 200
r = 0.03

CPPI(S_sim, V0, Gt, r, m=3)




    
    
    
    



