 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:30:17 2024

"""


import numpy as np
import numpy.random as rd
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy.optimize as opt


#%% Load Price Data


# Adjusted Closing Prices, corrected for any divididend price drops
prices = pd.read_csv("dji_close_2014_2024.csv", index_col=0)

prices = prices.loc["2021-01-01":]      
prices = prices / prices.iloc[0]        # Set all prices at beginning to 1

plt.plot(prices, "k", alpha=0.4)        # Paths of all stocks
plt.plot(prices["^DJI"], "r")           # Path of Index


#%% Returns & Stylized Facts

# Extract returns
rEmp = np.log(prices / prices.shift(1)).dropna(axis=0)
mu_hist = rEmp.mean() * 250 # Annualized (="per year") mean return
vola_hist = rEmp.std() * np.sqrt(250)

# third moment, typically negative skewness due to negative surprises
skew_hist = scs.skew(rEmp)    

# Fatter tails than normal distribution, leptokurtic typical for stocks
kurt_hist = scs.kurtosis(rEmp)  
# outliers appear more frequently than expected
# only the index is close to the normal distribution
# and also portfolios are closer to normal distribution
    

#%% CAPM

# Volatility can be taken from historic data (better from recent data)
# Also correlations can be used from historic data
# Need model for expected returns, e.g. CAPM

# Riskfree + (Risk Premium) * beta (cannot be diversified)
# Beta = Co-movement with market
# Use regression of stock returns on market returns
# Beta_i = covariance_with_market / sigma_M^2

def CAPM_estimator(rM, rA, E_rM=0.1, rS=.01, dt = 1/250):
           
    rComb = pd.concat((rM, rA), axis=1)   
    Cov = rComb.cov() / dt
    beta = Cov.iloc[0] / Cov.iloc[0,0]   # sigma_{M, i} / sigma_{M,M}
    ER = rS + (E_rM-rS) * beta  # expected returns
    
    return ER.iloc[1:], Cov.iloc[1:,1:], beta.iloc[1:]  # estimate assets' expected returns, covariances and bets


i = np.array([1,2,3])
rM = rEmp["^DJI"]     # daily return of market portfolio
rA = rEmp.iloc[:, i]   # daily returns of assets for our portfolio


mu, S, beta = CAPM_estimator(rM, rA)
vola = np.sqrt(np.diag(S))    # volatilties = square roots of variances (=diagonal of covariance matrix)


#%% Random Weights Portfolios & Minimum Variance Portfolio

N = len(mu)

# Show risk diversification, works good for many assets, portfolio always better risk-return ratio than individual stocks

for _ in range(1000):
    
    w = rd.rand(N)
    w = w / w.sum()   # portfolio weights, summing up to 1
    
    rP = mu @ w                  # portfolio expected return
    vP = np.sqrt(w.T @ S @ w)    # volatility
    
    plt.plot(vP, rP, "c.")     # Plot the random portfolio
  
    
plt.plot(vola, mu, "ok")   # Plot the variances and returns


xMVP = np.linalg.inv(S).sum(1) / np.linalg.inv(S).sum()    # slide 25

rP, vP = mu @ xMVP, np.sqrt(xMVP @ S @ xMVP)
plt.plot(vP, rP, "ro")

print(vP)
print(xMVP)


#%% Markowitz 


def MarkowitzOpt(mu, S, gamma=1):
    
    rP = lambda w: w @ mu
    vP = lambda w: np.sqrt( w @ S @ w)
    OF = lambda w: vP(w) * gamma - rP(w) * (1-gamma)   
    
    N = len(mu)
    
    cons = ({"type": "eq", "fun": lambda w: np.sum(w)-1}) # equality constraint: (sum_weights - 1) = 0
    nonneg = [(0, None) for i in range(N)] # for all assets i: 0 < w_i < infinity

    x0 = np.ones(N)/N
    result = opt.minimize(OF, x0, constraints=cons, bounds = nonneg)
    
    xOpt = result.x
    rOpt, vOpt = xOpt @ mu, np.sqrt(xOpt @ S @ xOpt)
    
    return xOpt, rOpt, vOpt
    

results = []

for gamma in np.linspace(0, 1, 51):
    
    xOpt, rOpt, vOpt = MarkowitzOpt(mu, S, gamma)
    results.append([vOpt, rOpt])
    
results = np.array(results)    
plt.plot(results[:,0], results[:,1], "-b")    # Markowitz Efficient Frontier


#%% Simulating Portfolio Returns


def return_Simulator(mu, S, nExp=10000, T=1):
    
    N = len(mu)
    sig = np.linalg.cholesky(S)   # square root of covariance matrix, meaning if sig is multiplied with itself, give cov-matrix
    
    # Built in also correlation from the assets in i
    rSim = rd.randn(nExp, N) @ (sig.T * np.sqrt(T)) + np.array(mu) * T
    
    return rSim


xM, rM, vM = MarkowitzOpt(mu, S, gamma=1)

# Now let's assess the performance with optimal weights of Markowiwtz using the simulator
rSim = return_Simulator(mu, S, nExp=10000) @ xM
rS, vS = rSim.mean(), rSim.std()

print(f"Markowitz: {rM, vM}")
print(f"Simulation: {rS, vS}")

# Will end up -40% up to +60%, meaning of volatility: deviations are possible
_ = plt.hist(rSim, 51)





#%% Block Bootstrapping (taking blocks out of time series)


def bootstrap(R, nSamp, b=1):
    
    N = R.shape[0]
    
    b = min(N-2, b) # should not crash, if we want to sample less than block lengths
    
    I = rd.randint(0, N-b, (nSamp//b+1,1) ) + np.arange(b)  # subsequent values

    I = I.flatten()[:nSamp]
    
    try: # when R is a dataframe:
        
        BS = R.iloc[I,:]
        BS.reset_index(drop=True, inplace=True)
        
    except: # otherwise; when R is an array
    
        BS = [R[i] for i in I]
        BS = np.array(BS)
        
    return BS


R = np.arange(101,141)
bootstrap(R, 39, 3)


#%% Experiments


# Bootstrap from empirical return and check the beta for bootstrapped sample
# Check how stable beta is (long bootstrap and long samples should be stable)
# If we vary the bootstrap, we will get different parameters (betas)
# Compare to yahoo finance betas (very different)

rM = rEmp["^DJI"]            # daily return of market portfolio
rA = rEmp.iloc[:, [1,2,3]]   # daily returns of assets for our portfolio

ER, Cov, beta    = CAPM_estimator(rM, rA)

bootstrapped = bootstrap(rEmp, 1000000, b=20)
rMB = bootstrapped["^DJI"]
rAB = bootstrapped.iloc[:, [1,2,3]]

BS_ER, BS_Cov, BS_beta = CAPM_estimator(rMB, rAB)

print(beta)
print(BS_beta)




