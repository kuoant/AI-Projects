#%% required packages and modules

import numpy as np
import numpy.random as rd
import pandas as pd



#%% retrieve returns
# ============================================================================

def readReturns(fn='dji_close_2014_2024.csv', 
                start='1900-01-01', before='2099-12-31',
                fromPrices=True, indAssets = None):

    # load returns from csv with empirical prices or returns for provided interval
    # column 0: market index; columns 1, 2, ...: assets
 
    data = pd.read_csv(fn, index_col=0 )
    data = data.loc[start:before]
    if fromPrices:  # if data (csv-file) are prices: 
        data = np.log(data/data.shift(1))  # extract log-returns from prices
        data = data.drop(data.index[0])    # first row contains NaN => drop

    rM = data.iloc[:,0]
    if indAssets is None:
        rA = data.iloc[:,1:]
    else:
        rA = data.iloc[:,indAssets]
    return rM, rA



#%% data generators & support functions
# ============================================================================

def CAPM_estimator(rM, rA, E_rM=0.1, rS = .03, dt = 1/250):  
    """ parameters, according to CAPM
        input:
            . market returns (rM) and asset returns (rA) 
            . frequency (dt; default: daily)
            . expected Market return (E_rM), risk free rate (rS), 
        returns: 
            annualized expected returns, covariances, and betas, based
    """
    
    rComb = pd.concat( (rM, rA), axis=1) # concatenate rM and rA into 1 dataframe
    Cov = rComb.cov()   # covariances for returns with period length dt
    Cov = Cov / dt      # annualize covariances 
    beta = Cov.iloc[0] / Cov.iloc[0,0]   # sigma_{M,i} / sigma_{M,M}
    ER = rS + (E_rM-rS) * beta  # expected returns
    E_r  = ER.iloc[1:]
    Cov  = Cov.iloc[1:,1:]
    beta = beta.iloc[1:]
    rA_filter = rA - rA.mean() + E_r*dt
    return E_r, Cov, beta, rA_filter


def returns_Simulator(E_r, Cov, dt=1/250, nSamp=250,**kargs):
    # simulate normally distributed returns 
    # E_r, Cov ... annualized  expected returns and covariances of assets
    # dt       ... length of time-steps (default: daily)
    # nSamp    ... number of samples (default: 250 for one year)
    try: # for multivariate data
        nAssets     = len(E_r)  # how many assets?
        sig         = np.linalg.cholesky(Cov)   # "square root" of covariance
        rSim        = rd.randn(nSamp, nAssets) @ (sig.T * np.sqrt(dt)) + np.array(E_r)*dt
    except: # univariate
        rSim = rd.randn( nSamp,1 ) * np.sqrt(Cov*dt) + E_r*dt
    return rSim


def bootstrap(rEmp, nSamp=250, b=1,**kargs):
    # block-bootstrap for nSamp samples and block-length b
    N = rEmp.shape[0]     # number of available observations
    b = min(N-2, b)    # ensure block length does not exceed number of observations
    if b<2: # no actual blocks, just individual observations
        I = rd.randint(0, N-b+1, nSamp )
    else: # blocks of b subsequent obersvations
        nBlockSamp = nSamp//b+1  # required number of blocks 
        I = rd.randint(0, N-b+1, nBlockSamp )  # indices for first observation for the blocks, e.g., i, s.t.  0 ≤ i < N
        I = I.reshape((nBlockSamp,1)) + np.arange(b)  # using "broadcasting": extend i to [i+0, i+1, ..., i+(b-1)] according to blocklength
        I = I.flatten()[:nSamp]

    try: # assume R is a dataframe:
        BS = rEmp.iloc[I,:]
        BS.reset_index(drop=True, inplace=True)
    except: # otherwise; when R is an array or a list
        BS = [ rEmp[i]  for i in I ]
        BS = np.array(BS)    

    return BS


def simulatePaths(method, S0=1., dt=1/250, T=1., **kargs):
    """ create price paths
    
    methods: 
        'GBM' (geometric Brownian motion): requires E_r, Cov
        'bootstrap': requires rEmp, b; dt
        otherwise: use historical sample; requires rEmp
     T: overall time horizon (default: 1 year)
    """

    nSamp = int(T//dt)

    match method.upper():
        case 'GBM':  #  geometric Brownian motion; requires E_r=.., Cov=...
            rSim  = returns_Simulator( dt=dt, nSamp=nSamp,**kargs)

        case 'BOOT' | 'BOOTSTRAP' : # block bootstrap; assumption: original observations' frequency corresponds to dt
            rSim  = bootstrap( nSamp=nSamp, **kargs )
            rSim  = np.array(rSim)
        
        case 'EMP' | 'HIST' | 'HS' : # use historical (empirical) return series
            rSim = np.array(kargs['rEmp'])
        case _ :  # for all other situations: keep prices at original level
            rSim = np.zeros((nSamp,1))
    
    r0 = np.array(0*rSim[0])
    rSim = np.vstack((r0, rSim)) # insert a leading row with 0 to retain S0 
    price = S0 * np.exp( np.cumsum( rSim, axis=0 ) )
  
    return price



#%% "PORTFOLIO" CLASS
# ============================================================================


class portfolio():
    """
    initialization:
        S0 ... initial price of risky asset(s)
        V0 ... initial wealth
        rS ... riskfree interest rate (for cash; annualized)
        dt ... time increments
        wOpt ... initial asset weights (relative to entire value; default: 0)
    
    attributes:
        cash  ... current cash position
        price ... current prices
        quant ... quantity of stocks
        
    properties:
        risky ... value of risky assets
        value ... total value: risky + cash
        wCurr ... current weights of risky assets (relative to entire value)
    
    methods:
        updatePrice(S_t)     
        buySell(deltaQuant) ... number of stocks to buy (negative: sell)
        rebalance(wOpt)     ... target portfolio weights (relative to entire value)
    """
    

    def __init__(self, S0, V0=1., rS=0.03, dt=1/250, wOpt=0.):
        if not hasattr(S0,'__iter__'): 
            S0 = [S0]
        self.S0     = np.array(S0)
        self.V0     = float(V0)
        self.cash   = self.V0
        self.price  = np.array(S0)
        self.quant  = np.zeros_like(S0)
        
        self.rS     = rS  
        self.dt     = dt 
        
        self.rebalance(wOpt)
        
    def __str__(self):
        label = ['risky', 'cash', 'total']
        value = [f'{v:,.2f}' for v in [self.risky, self.cash, self.value]]
        lines = [f'{v:>15}  {l}' for v,l in zip(value, label)]
        lines.insert(2,'='*25)
        weights = f'weights: {self.wCurr.round(3)} + {(self.cash/self.value).round(3)}'
        lines = [weights] + lines
        position = '\n'.join(lines)
        return position

    def updatePrice(self, S):
        self.price[:] = S
        self.cash = self.cash * np.exp( self.rS * self.dt )

    @property
    def risky(self):
        
        return self.price @ self.quant
        
        # <<<<<<<<<<<<<<<<<<<<<    YOUR CODE HERE    >>>>>>>>>>>>>>>>>>>>>>>>

    @property
    def value(self):
        
        return self.cash + self.risky
        
        # <<<<<<<<<<<<<<<<<<<<<    YOUR CODE HERE    >>>>>>>>>>>>>>>>>>>>>>>>
    
    @property
    def wCurr(self):
        
        w = self.quant * self.price / self.value
        
        return w
        
        # <<<<<<<<<<<<<<<<<<<<<    YOUR CODE HERE    >>>>>>>>>>>>>>>>>>>>>>>>
        
    def buySell(self, deltaQ):
        
        self.quant = self.quant + deltaQ # deltaQ is a vector here [10,0,0] = buy 10 of a
        self.cash = self.cash - deltaQ @ self.price
        
        
        # <<<<<<<<<<<<<<<<<<<<<    YOUR CODE HERE    >>>>>>>>>>>>>>>>>>>>>>>>

    def rebalance(self, wOpt):
        
        wOpt = np.array(wOpt)
        quantOpt = (self.value * wOpt) / self.price
    
        deltaQ = quantOpt - self.quant     
        self.buySell(deltaQ)        
        # Sell everything is pf.rebalance([0])
        
        
        # <<<<<<<<<<<<<<<<<<<<<    YOUR CODE HERE    >>>>>>>>>>>>>>>>>>>>>>>>

