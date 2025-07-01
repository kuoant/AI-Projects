"""
support functions for Numerical Optimization and Simulation
version 2.0 (2024-02-16)

NUMERICAL ROUTINES:
===================
    deriv(f,x0,ord=1,h=2**(-5))
    Gradient(f,x, h=2**(-5))
    Hessian(f,x,h=2**(-5))


VISUALISATION:
==============
    funplot2d(f, x=[-5,5])
    funplot3d(f, x1=[-5,5], x2=[-5,5],type='surf'):
        simple function plot
        x1:  2 elements: => lower/upper bound for range, 
            >2 elements: values
        x2: None => univariate function, otherwise like x1
        type: 'surf' for surface, 'cont' for contour
    
    pointplot(x, y, col=None) : 
        x.shape = (n,), y.shape = (n,) : 2d scatterplot
        x.shape = (n, 2), y=None       : 2d scatterplot
        x.shape = (n,2),  y.shape = (n,) : y .. markersize
        col = 'iter' : colorgradient based on index 
            
    Taylorplot(f, x0, s=[-1,1], order=1, fmt=None)


SUPPORT FUNCTIONS FOR "ACCOMPANYING PROBLEMS":
==============================================
    Problem 1: univariate 
        fu01 ... fu06
        
    Problem 2: multivariate 
        fm11 ... fm14
        fm21 = rosenbrock(x)
        fm22 = trid(x)
        fm23 = booth(x)
        fm24 = branin(x)
        fm25 = styblinski_tang(x)
        fm26 = levy(x)
        fm27 = ackley(x)
        
    Problem 3: binary 
        fb31, fb32, fb33

    Problem 4:
        profitDS(p, comb='aaa') 
        
""" 



import numpy as np
import matplotlib.pyplot as plt


#%% general support functions: 
# ======================================================================
    
#%% numerical routines

#----------------------------------------------------------
def deriv(f, x0, ord=1, h=2**(-5)):
#----------------------------------------------------------
    # numerical derivation with central differences
    if ord<=0:
        return f(x0)
    else:
        fp = deriv(f,x0+h,ord=ord-1,h=h)
        fm = deriv(f,x0-h,ord=ord-1,h=h)
        return (fp-fm)/(2*h)


#----------------------------------------------------------
def Gradient(f, x, h=2**(-5)):
#----------------------------------------------------------
    d = len(x)
    s = np.zeros((d,))
    J = np.zeros((d,))
    for i in range(len(x)):
        s[i] = h
        J[i] = (f(x+s) - f(x-s))/(2*h)
        s[i] = 0
    return J


#----------------------------------------------------------
def Hessian(f, x, h=2**(-5)):
#----------------------------------------------------------
    d = len(x)
    s = np.zeros((d,))
    v = np.zeros((d,))
    H = np.zeros((d,d))
    foo = f(x)
    for i in range(d):
        s[i] = h
        H[i,i] = (f(x+s) + f(x-s) - 2*foo)/(h**2)
        for j in range(i):
            v[j] = h
            H[i,j] = (f(x+s+v) + f(x-s-v)-f(x+s-v)-f(x-s+v))/(4*h**2)
            H[j,i] = H[i,j]
            v[j] = 0
        s[i] = 0
    return H



#%% visualisation
# ======================================================================

#----------------------------------------------------------
def funplot2d(f, x=[-5,5], *args, **kargs):
#----------------------------------------------------------
    """ 
    simple function plot
    x:  2 elements: => lower/upper bound for range, 
        >2 elements: values
    type: 'surf' for surface, 'cont' for contour
    """
    if len(x)==2:
        X = np.linspace(x[0],x[1], 101)
    else:
        X = np.array(x)
    Y = np.vectorize(f)(X)
    
    plt.plot(X,Y)
    plt.xlabel('x')
    plt.xlabel('y')


#----------------------------------------------------------
def funplot3d(f, x=[[-5,5], [-5,5]], type='surf', nGrid=31, *args, **kargs):
#----------------------------------------------------------
    X1, X2 = np.linspace(*x[0], nGrid), np.linspace(*x[1], nGrid)
    Y = np.array([[f(np.array([x1,x2])) for x1 in X1] for x2 in X2])
    
    if type=='surf':
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        MX,MY = np.meshgrid(X1,X2)
        _ = ax.plot_surface(MX, MY, Y, alpha=.5)
    else:
        plt.contourf(X1,X2,Y, alpha=.7, cmap='coolwarm')
        plt.colorbar()



#----------------------------------------------------------
def pointplot(x, y=None, col=None):
#----------------------------------------------------------
    if col=='iter':
        col = np.linspace(0,1,len(y))
        
    if y is None:
        plt.scatter(x[:,0], x[:,1], c=col)  
    if x.shape == y.shape: # 1d plot
        plt.scatter(x,y,c=col)
    else:
        ax = plt.gca()
        ax.scatter(x[:,0], x[:,1], y,c=col)    



#----------------------------------------------------------
def Taylorplot(f, x0, s=[-2,2], order=1, fmt=None):
#----------------------------------------------------------
    factorial = lambda n: np.prod(np.arange(1,n+1))
    if order>1:
        s = np.linspace(s[0],s[1],101)
    else:
        s = np.array(s)
    f_hat = f(x0) 
    for o in range(1,order+1):
        f_hat = f_hat + deriv(f,x0,o)*s**o/factorial(o)
    if fmt is None:
        plt.plot(x0+s,f_hat,linewidth=1)
    else:
        plt.plot(x0+s, f_hat, fmt, linewidth=1)





#%% ======================================================================
#/%          support for problem set (accompanying problems)
#/% ======================================================================


#%% continuous univariate functions

fu01 = lambda x: x**2 - 5*x
fu02 = lambda x: x**3 - 5*x
fu03 = lambda x: -np.exp(-(x-2)**2)
fu04 = lambda x: np.abs(x-3)
fu05 = lambda x, b=0.1: np.abs(x-3) + b*np.sin(np.pi*x)
fu06 = lambda x, b=0: np.abs(x)**1.5 + 0.9 * np.abs(4-x**2)**0.5 * np.sin(b*np.pi*x)

#%% continuous multivariate functions

#----------------------------------------------------------
def fm11(x): 
    return np.sum((x-np.arange(len(x)))**2)

#----------------------------------------------------------
def fm12(x):
    d = len(x)
    f = 0.0
    for i in range(d):
        for j in range(d):
            f += (x[i]-(i+1))*(x[j]-(j+1))/(1+(i-j)**4)
    return f


#----------------------------------------------------------
def fm13(x,Q,c):
    return x@Q@x + x@c

    
#----------------------------------------------------------
def fm14(x):
    return np.mean(np.abs(x)**4)


#----------------------------------------------------------
def rosenbrock(x):
    i = np.arange(len(x)-1).astype('int')
    return np.sum(100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2)

fm21 = rosenbrock


#----------------------------------------------------------
def trid(x):
    return np.sum((x-1)**2) - np.sum(x[1:]*x[:-1])

fm22 = trid


#----------------------------------------------------------
def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] -5)**2

fm23 = booth


#----------------------------------------------------------
def branin(x):
    a,b,c = 1, 5.1/(4*np.pi**2), 5/np.pi
    r,s,t = 6.0, 10.0, 1/(8*np.pi)
    return a*(x[1]-b*x[0]**2 + c*x[0]-r)**2 + s*(1-t)*np.cos(x[0]) + s    

fm24 = branin


#----------------------------------------------------------
def styblinski_tang(x):
    return np.sum(x**4-16*x**2-5*x)/2

fm25 = styblinski_tang


#----------------------------------------------------------
def levy(x):
    w = 1+(x-1)/4
    wi,wd = w[:-1], w[-1]
    f  = np.sin(np.pi*w[0])**2
    f += np.sum((wi-1)**2 * (1 + 10 * np.sin(np.pi*wi +1)**2)) 
    f += (wd-1)**2 * (1+np.sin(2*np.pi*wd)**2) 
    return f

fm26 = levy


#----------------------------------------------------------
def ackley(x):
    a,b,c = 20.0, 0.2, 2*np.pi
    f  = -a*np.exp(-b*np.sqrt(np.mean(x**2)))
    f -= np.exp(np.mean(np.cos(c*x)))
    f += a + np.exp(1)
    return f

fm27 = ackley



#%% binary problems

#----------------------------------------------------------
fb31 = lambda x: np.sum(x)

fb32 = lambda x: x @ (2**(np.arange(len(x)-1,-1,-1)))

fb33 = lambda x: np.sum(np.cumprod(x))



#%% Profit and Loss

#----------------------------------------------------------
@np.vectorize
def profitDS(p,comb='aaa'):
#----------------------------------------------------------
    # supply function:
    if p<=0:
        p=0.01
    D,S,C = comb
    if D=='a':
       qD = 6000-1500*p
       qD = np.maximum(qD,0)
    elif D=='b':
        qD = 100 * np.exp(-p)
    else:
        qD = 5000 * 1.2**(-p) + 7000*p**(-1.3)
        #         qD = 60 * np.exp(-0.9*p) + 40 * np.exp(-1.1*p)

    
    if S=='a':
        qS = qD
    elif S=='b':
        qS = np.minimum(qD,2000*p)
    else:
        qS = np.minimum(qD,10000)
    
    if C=='a':
        cS = 0
    elif C=='b':
        cS = qS
    elif C=='c':
        cS = np.sqrt(qS)
    else:
        cS = 0.3 * qS ** 0.9
        
    return qS*p - cS
        
       

       

