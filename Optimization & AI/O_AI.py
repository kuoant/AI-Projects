#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:10:38 2024

"""

import OptAI_support_24F as oai
import matplotlib.pyplot as plt


#%% Bisection for Minimization

# simple, but good method, only for univariate functions
# works only if we have a minimum inbetween R, border-solution otherwise
def Bisection_min(f, R, nIter=100, tol=1e-7):
    
    a,b = R
    
    for it in range(nIter):
        
        c = (a+b) / 2      # center for interval
        df_c = oai.deriv(f, c) # f'(x_c)
        
        if abs(df_c) < tol: # close enought to 0
            break
        elif df_c > 0:
            b = c  # we look at minimization, that's why we must use b=c!
        else:
            a = c  # if we look at maximization we have to reverse a=c and b=c
            
    c = (a+b)/2
    return c


f = lambda x: oai.fu06(x, b=3)
R = [-4,16]
x_opt = Bisection_min(f, R, tol=1e-19)
f_opt = f(x_opt)
print(x_opt, f_opt, oai.deriv(f, x_opt))

oai.funplot2d(f, R)
oai.Taylorplot(f, x_opt)
plt.plot(x_opt, f_opt, "ro")


#%% Gradient Search for Minimization

# Idea is good, but calibration of a is difficult
def Gradient_min(f, x0, nIter=100, thresh=1e-9, a=0.5):
    
    for it in range(nIter):
        
        s = - a * oai.deriv(f, x0)
        
        if abs(s) < thresh:
            break
        
        x0 = x0 + s
        
        print(it)
        
    return x0


f = lambda x: oai.fu06(x, b=2)
x_opt = Gradient_min(f, x0=-1, a=0.02, nIter=1000)
print(x_opt, f(x_opt), oai.deriv(f, x_opt))

# Get stuck in local optimum
oai.funplot2d(f)
plt.plot(x_opt, f(x_opt), "ro")
oai.Taylorplot(f, x_opt)


#%% Newton


def Newton(f, x0, nIter=100, thresh=1e-9):
    
    for it in range(nIter):
        
        s = - oai.deriv(f, x0) / oai.deriv(f, x0, ord=2)
        
        if abs(s) < thresh:
            break
        
        x0 = x0 + s
        
    return x0


f = lambda x: oai.fu06(x, b=2)
x_opt = Newton(f,20)
f_opt = f(x_opt)
print(x_opt, f_opt)

# We cannot find the minimum, because the derivative does not exist
oai.funplot2d(f, R)
oai.Taylorplot(f, x_opt)
plt.plot(x_opt, f_opt, "ro")

        
#%% Nelder-Mead

import scipy.optimize as opt
        
f = oai.fm27
x0 = [3,3]
res = opt.minimize(f, x0, method="Nelder-Mead")
oai.funplot3d(f) 
print(res) 
# Success true, if we did not run out of iterations
# Rosenbrock (fm21) is not considered to be good for gradient-based method
# Nelder-Mead do not acknowledge different optimum, we have to experiment 
# Convex problems: go for gradient-based methods
# Quasi-convex problems: go for gradient-free Nealder-Mead



#%% Linear Programming

import scipy.optimize as opt

opt.linprog(c=[-20, -30], A_ub=[[1,0],[0,2],[3,2]], b_ub=[4,12,18])



#%% QP

import numpy as np
import scipy.optimize as opt

Q = [[5,2], [2,7]]

fun = lambda x: x.T @ Q @ x

x = np.array([1,4])


opt.minimize(fun, x)
    


#%% Knapsack Problem Complexity

for n in range(1,20):
    print(n, 2**n)

#%% Travelling Salesman Complexity

def fact(n):
    f = 1
    for i in range(1, 1+n):
        f *= i
        
    return f

for i in range(1,20):
    print(i, fact(i))

#%% Project Selection (Greedy algorithm)

import numpy as np

c = [20, 300, 400, 400, 450, 900]

p = [1, 90, 90, 75, 100, 150]

roi = np.array(p) / np.array(c)

index = np.argsort(-roi)

np.cumsum(np.array(c)[index])


#%% Monte Carlo Method

# for univariate functions
def MonteCarlo_min(f, R, N=100):
    
    xCand = R[0] + rd.rand(N) * (R[-1]-R[0])   # Scaling the random number
    fCand = [f(x) for x in xCand]
    
    iBest = np.argmin(fCand)
    xOpt = xCand[iBest]
    
    plt.plot(xCand, fCand, ".")
    plt.plot(xOpt, f(xOpt), "or")
    
    return xOpt


# Convergence: the more experiments we run, we get closer to true solution
# without being distorted, and we often are faster
f = lambda x: oai.fu05(x, b=2)
oai.funplot2d(f,R)

MonteCarlo_min(f, [-10, 15])
# Now we have the chance to solve also problems with local minimums
# Problem: need high computational power, if we increase points
    

# Generate uniform u in range [0, 1] random number with following logic:
# Scaling by R[0] + rd.rand(N) * (R[-1]-R[0])



#%% SDE

# Same as gradient search but with noise to prevent premature convergence
# Avoid systematic noise, expecation of noise should be 0

# Scale down sd of normal distribution as sd=1 might be a bit too much
# Calibrate s_noise parameter is crucial for the method

#  Just adding noise yields random outcomes, need to check for improvement.

def SDE_min(f, x0, nIter=100, a=0, s_noise=0.4):
    
    test = []
    best = []
    
    xC, fC = x0, f(x0)
    
    for it in range(nIter):
        
        s = - a * oai.deriv(f, xC) + rd.randn() * s_noise
        xN = xC + s
        fN = f(xN)
        
        if fN < fC:
            xC, fC = xN, fN
            
            best.append(xN)
            
        test.append(xN)
        
    return xC, test, best
        


f = lambda x: oai.fu06(x, 15)

oai.funplot2d(f)
xOpt, test, best = SDE_min(f, 3)

plt.plot(test, f(np.array(test)), "ok")
plt.plot(best, f(np.array(best)), "og")
plt.plot(xOpt, f(xOpt), "oy")

print(xOpt)
        

#%% Threshold Acceptance for Univariate Problems

# Now we completely delete the gradient, as it is not necessary.
# Threshold which is decreasing over time in order to allow for uphill.

# But now we want also to track the acting optimum = best solution so far.

def TA_min(f, x0, nIter=10, s_noise=1., Thresh=10):
    
    test = []
    best = []
    
    xC, fC = x0, f(x0)
    xOpt, fOpt = xC, fC        # initial acting optimum = best solution so far
    
    deltaThresh = Thresh / nIter  # Introduce variable Threshold
    
    for it in range(nIter):
        
        s = rd.randn() * s_noise  # if uniform noise we need to decuct 0.5: rd.randn() - 0.5 to get mean=0
        xN = xC + s
        fN = f(xN)
        test.append(xN)
        
        if fN - fC < Thresh:   # Accept Uphill if it is "not too bad"
            xC, fC = xN, fN
            
        if fN < fOpt:     # is the new candidate the new acting optimum?
            xOpt, fOpt = xN, fN
            best.append(xN)
        
        Thresh -= deltaThresh   # make threshold stricter
        
        
    return xOpt, np.array(test), np.array(best)



f = lambda x: oai.fu06(x, 15)

oai.funplot2d(f, [-15, 15])
    
xOpt, test, best = TA_min(f, 15)

plt.plot(test, f(test), "ok")
plt.plot(best, f(best), "go")
plt.plot(xOpt, f(xOpt), "or")
print(xOpt)

        
        
        
        
#%% Threshold Acceptance for Multivariate Problems
        

# TA goes under Markov Chain Monte Carlo, because we improve upon existing candidates
def TA_min(f, x0, nIter=100000, s_noise=2., Thresh=20):

    # Initial Point needs to be a np.array
    if not hasattr(x0, "__iter__"):  # already list or array?
        x0 = [x0]
        
    x0 = np.array(x0)
    
    D = len(x0)  # how many dimensions (=elements)?    
    
    # Initialize search
    xC, fC = x0, f(x0)
    xOpt, fOpt = xC, fC        # initial acting optimum = best solution so far
    
    deltaThresh = Thresh / nIter  # Introduce variable Threshold
    
    # perform search
    for it in range(nIter):
        
        s = rd.randn(D) * s_noise  # if uniform noise we need to decuct 0.5: rd.randn() - 0.5 to get mean=0
        xN = xC + s
        fN = f(xN)
        
        if fN - fC < Thresh:   # Accept Uphill if it is "not too bad"
            xC, fC = xN, fN
            
        if fN < fOpt:     # is the new candidate the new acting optimum?
            xOpt, fOpt = xN, fN
        
        Thresh -= deltaThresh   # make threshold stricter
        
    return xOpt



f = lambda x: oai.fm27(x)
oai.funplot3d(f)

xOpt = TA_min(f, [-3,-3], 10000, s_noise=5, Thresh=3)

print(xOpt)



#%% Random Walk

T = 1000
D = 2
x = np.zeros((T,D))

for t in range(T-1):
    x[t+1] = x[t] + rd.randn(D)
        
plt.plot(x[:, 0], x[:,1], "o-")


#%% Genetic Algorithm for maximization


import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import OptAI_support_24F as oai


# Gold-standard for binary/integer problems
def GA_max(f, D,

    # means we have 50 generations, if function evaluation=1000 and popSize=20
    
    FE = 2000,     # number of function evluations in total
    PopSize = 10,  # population size
    p_XO = 0.5,    # probability for cross-over: prob from parent 2
    p_mut = 0.1    # probability that a bit is flipped
    ):

    # Initialize 1st Generation
    xC = (rd.rand(PopSize, D) < 0.5).astype(int)  # current population
    fC = np.array([f(x) for x in xC])             # current individuals fitness 
    
    # Determine 1st Elitist
    iE = np.argmax(fC)              # who is the elitist (=best candidate)?
    xE, fE = xC[iE].copy(), fC[iE]  # record its solution and objective function
    
    # Iterate over generations
    nGen = FE // PopSize - 1
    for gen in range(nGen):
        
        # Crossover
        p1 = np.arange(PopSize)
        p2 = rd.permutation(PopSize)             # permutation gives us a random numbers to pair with parent 1
        from_p1 = rd.rand(PopSize, D) < p_XO     # which bits come from parent 1
        xN = np.where(from_p1, xC[p1], xC[p2])   # new candidates from cross-over
        
        # Mutation
        mut = rd.rand(PopSize, D) < p_mut     # which bits to mutate?
        xN[mut] = 1 - xN[mut]                 # mutate 0->1 and 1->0    
        
        # Evaluate offspring
        fN = np.array([f(x) for x in xN])  # new individuals fitness    
        
        # 1-1 tournament to find survivors
        replace = fN > fC           # does it outperform parent 1?
        xC[replace] = xN[replace]   # update population
        fC[replace] = fN[replace]   # update their fitness value (=objective values)
        
        # Determine new elitist
        iE = np.argmax(fC)
        
        if fC[iE] > fE: # do we have a new elitist?
            xE, fE = xC[iE].copy(), fC[iE]    
            
    return xE


# for all fb* functions the optimal solution is an x array of ones
# oai.fb31 just counts the 1 in the x-array
# oai.fb32 binary value of the x-arrray
# oai.fb33 how many leading ones we have, e.g. x=[1,1,1,0,0] would give 3

f = oai.fb33   # objective function
D = 32         # dimensions: length of bit-string

# this means we could have very, very many combinations, but only
# 2000 function evaluations are needed thanks to genetic algorithms

xOpt = GA_max(f, D)
print(xOpt)

#%% Example 1.6 Project Selection Problem solved with GA

p = [1, 90, 90, 75, 100, 150]
c = [20, 300, 400, 400, 450, 900]
B = 1000

# 3 Ways to deal with constraints
# 1) ensure new candidate is by construction is correct
# 2) repair function, infeasible solution turned into solution
# 3) punishment function, whenever we exceed budget

profit = lambda x: np.sum(x * p)
cost   = lambda x: np.sum(x * c)
punish = lambda x: (cost(x)>B) * 100 + np.max( [cost(x)-B, 0] ) * 20 # specific
f      = lambda x: profit(x) - punish(x)

D = len(p) # how many projects?
xOpt = GA_max(f, D)
print(f"{xOpt} to earn {profit(xOpt)}, costing {cost(xOpt)}")
print(xOpt)



#%% Differential Evolution for maximization

import numpy as np
import numpy.random as rd
import OptAI_support_24F as oai
import matplotlib.pyplot as plt


def DE_max(f, D, 
           R = [-10, 10],  # range for initial positions
           FE = 10000, 
           popSize = 30, 
           F=0.7,  # scaler: weight for difference vector, usually 0.7
           pXO = .3 # cross-over probability
           ):
    
    
    # Initialize 1st Generation
    xC = rd.rand(popSize, D) * (R[1] - R[0]) + R[0] # initalize random points
    fC = np.array([f(x) for x in xC])
    
    # Evolution
    nGen = FE // popSize - 1
    for gen in range(nGen):
            
        # Linear combination, equivalent to mutation 
        # intially they are far apart, more exploration
        p1, p2, p3 = [rd.permutation(popSize) for _ in range(3)] # permutate 
        xN = xC[p1] + F * (xC[p2] - xC[p3])
        
        # Crossover XO with 4th parent
        XO = rd.rand(popSize, D) < pXO  # with probXO take lin comb, else current
        xN = np.where(XO, xN, xC)
    
        # Evaluation
        fN = np.array([f(x) for x in xN])
        
        # 1-1 Tournament
        replace = fN > fC
        xC[replace] = xN[replace]
        fC[replace] = fN[replace]
        # print(gen, fC.mean()) # get better with every generation
        
        
    # Pick the elitist
    iE = np.argmax(fC) 
    
    return xC[iE]


f = lambda x: -oai.fm27(x)
oai.funplot3d(f)
plt.show()

FE = 2000
nExp = 10
psE = [5, 10, 15, 20, 30, 50, 100, 200]  # try for different population sizes
results = np.zeros((len(psE), nExp))

# check how close we are to the true solution
for i, popSize in enumerate(psE):
    for e in range(nExp):
        xOpt = DE_max(f, 2, FE = FE, popSize = popSize)
        results[i, e] = f(xOpt)

# sort them and plot for all populations to see how good the populations work
results = np.sort(results, 1)
plt.plot(results.T)
plt.legend(psE)




#%% Plotting fu01

# Convex function; one global, no local optima
f = oai.fu01
oai.funplot2d(f, [-4,8])

oai.Taylorplot(f, 5, order=2)

#%% Plotting fu02

f = oai.fu02

oai.funplot2d(f, x=[-4,6])

for x0 in [-3,-2,-1,0,1,2,3,4]:
     oai.Taylorplot(f, x0=x0, order=2, s=[-3,3])
     

#%% Plotting fu03

# Problematic 1st & 2nd derivatives in left/right direction misleading!
f = oai.fu03 
oai.funplot2d(f, x=[-2,6])


# Second derivative might be misleading, depending on where we are
# the Taylor expansion is both concave or convex
for x in [0,2,4]:
    oai.Taylorplot(f, x, order=1, s=[-1,1], fmt="g")
    oai.Taylorplot(f, x, order=2, s=[-1,1], fmt="r")


#%% Plotting fu04

# Derivative does not exist in x=3, and the 2nd derivative is throughout 0
# contains no information at this point
f = oai.fu04
oai.funplot2d(f, x=[-3,6])
plt.grid()

#%% Plotting fu05

# Now it's continuous again, but has many local optimums to get stuck
f = lambda x: oai.fu05(x, b=0.5)
oai.funplot2d(f, x =[-2,6])
plt.grid()

for x0 in [-1.5,0,1.6,3.28,5]:
    oai.Taylorplot(f, x0=x0, order=1, s=[-1,1], fmt="r")


#%% Plotting fu06
 
# Funny behavior depending on b, gradient-based methods lead to chaotic search
f = lambda x: oai.fu06(x,b=1.2)
oai.funplot2d(f, x =[-7.5, 7.5])
plt.grid()
    
for x0 in range(-7, 7, 1):
    oai.Taylorplot(f, x0=x0, order=1, s=[-0.5,0.5], fmt="r")











