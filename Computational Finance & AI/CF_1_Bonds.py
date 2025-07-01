#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:37:41 2024

"""

import numpy as np
import matplotlib.pyplot as plt


#%% Exercise 1.1 Terminal Value 

def terminal_value(V0, R, T, m=1):
        
    VT = V0 * (1+R/m)**(m*T)

    return VT

# get slightly more if payments are more frequent
V0 = 1000
R = 0.05
T = 1
m = 100

VT = terminal_value(V0, R, T=T, m=m)
print(VT)

#%% TV for different interest rates & 10 years

# Plot for 1 Interest Rate, 10 years

TSet = list(range(11))

# list comprehension:
VT = [ terminal_value(V0, R, T) for T in TSet]

plt.plot(TSet, VT, "o-")
plt.show()

# Different Interest Rates, 10 years

RSet = [0, 0.01, 0.05, 0.10]

for R in RSet:
    VT = [terminal_value(V0, R, T) for T in TSet]
    plt.plot(TSet, VT, "o-")
    
plt.legend(RSet)
plt.grid()




#%% Exercise 1.2 If m -> infinity

R = 0.05
T = 1
V0 = 1
M = [1, 2, 4, 8, 16, 32, 64, 128, 256]

V1 = [terminal_value(V0, R, T, m) for m in M]   
plt.plot(M, V1)

# result converges to:
Re = np.exp(R) - 1
print(Re + 1)

plt.axhline(y = 1 + Re, color="r")


#%% Exercise 1.6 Bond Pricing


def bond_price(F=100, c=0., T=1, y=0.):
    
    B0 = F / (1+y)**T
    
    if c > 0:
        for t in range(1,T+1): # for t = 1, 2, ..., T
            B0 += (F*c) / (1+y)**t
    
    return B0


T = 10
F = 100
c = 0.05
Y = [0.03, 0.05, 0.10]
t = np.arange(0,T+1)

for y in Y:
    
    B0 = bond_price(F, c, T, y)
    V_t = B0 * (1+y)**t   # wealth at t if coupons are re-invested at interest y
    plt.plot(t, V_t)
    print(f"yield = {y} ==> {B0}")

plt.legend(Y)

# Current wealth raises if we have low interests. But wealth
# grows much smaller because we do not have nice reinvestments opportunities.
# The duration is the point where all the future wealth are equal.



#%% Exercise 1.7 Wealth over time with Saving the Coupons


F = 100
c = 0.03
T = 10
y = 0.05
budget = 100_000

# Intially with savings account
p_bond = np.zeros(T+1)
p_bond[0] = bond_price(F, c=c, T=T, y=y)

n_bonds = np.zeros(T+1)
n_bonds[0] = budget / p_bond[0]

savings = np.zeros(T+1)


for t in range(1, T+1):
    coupons    = n_bonds[t-1] * (F*c)  # total coupon payments
    savings[t] = savings[t-1] * (1+y) + coupons # assume we y is paid as interest
    n_bonds[t] = n_bonds[t-1]
    p_bond[t]  = bond_price(F, c=c, T=T-t, y=y) # remaining time to matruity T-t

# at maturity:
wealth =  savings + n_bonds * p_bond

# both should be equal
print(wealth)
print(budget * (1+y)**T)

# both should be equal
plt.plot(wealth)
plt.plot(budget * (1+y)**np.arange(0,11))


#%% Exercise 1.7 Wealth over time with Reinvestment of Coupons in Bond


F = 100
c = 0.03
T = 10
y = 0.05
budget = 100_000


p_bond = np.zeros(T+1)
p_bond[0] = bond_price(F, c=c, T=T, y=y)

n_bonds = np.zeros(T+1)
n_bonds[0] = budget / p_bond[0]


for t in range(1, T+1):
    p_bond[t]  = bond_price(F, c=c, T=T-t, y=y) # remaining time to matruity T-t
    coupons    = n_bonds[t-1] * (F*c)  # total coupon payments
    n_bonds[t] = n_bonds[t-1] + coupons / p_bond[t]
 
# at maturity:
wealth = n_bonds * p_bond

# both should be equal
print(wealth)
print(budget * (1+y)**T)

# both should be equal
plt.plot(wealth)
plt.plot(budget * (1+y)**np.arange(0,11))


#%% Exercise 1.7 Wealth over time with Reinvestment of Coupons in Bond (only full amounts allowed)


F = 100
c = 0.03
T = 10
y = 0.05
budget = 100_000


p_bond = np.zeros(T+1)
p_bond[0] = bond_price(F, c=c, T=T, y=y)

n_bonds = np.zeros(T+1)
n_bonds[0] = np.floor(budget / p_bond[0])

savings = np.zeros(T+1)
savings[0]= budget - n_bonds[0] * p_bond[0]


for t in range(1, T+1):
    
    p_bond[t]  = bond_price(F, c=c, T=T-t, y=y) # remaining time to matruity T-t
    coupons    = n_bonds[t-1] * (F*c)  # total coupon payments
    delta_n    = np.floor(coupons / p_bond[t])
    n_bonds[t] = n_bonds[t-1] + delta_n
    savings[t] = savings[t-1] * (1+y) + coupons - delta_n * p_bond[t]
 
# at maturity:
wealth =  savings + n_bonds * p_bond

# both should be equal
print(wealth)
print(budget * (1+y)**T)

# both should be equal
plt.plot(wealth)
plt.plot(budget * (1+y)**np.arange(0,11))

# Idea in Bond Pricing: Whenever you receive coupons, you can buy more or less bonds





