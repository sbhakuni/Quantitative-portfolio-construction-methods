
import numpy as np 
import cvxopt
from numpy.linalg import inv
import pandas as pd
import yfinance as yf


def weights_equal(Sigma):
    n = Sigma.shape[1]
    w = np.ones((n))*1.0/n
    return w
    
def weights_mv(Mu, Sigma, delta=1.0, short_selling=False):
    n = Sigma.shape[1]
    Mu = np.asmatrix(Mu)
    Sigma = np.asmatrix(Sigma)
    if short_selling == False:
        obj = cvxopt.matrix(delta/2*Sigma)
        q = cvxopt.matrix(Mu, (n,1))
        A = cvxopt.matrix(1.0, (1,n))
        b = cvxopt.matrix(1.0)
        G = cvxopt.matrix(0.0, (n,n))
        G[::n+1] = -1.0
        h = cvxopt.matrix(0.0, (n,1))
        w = cvxopt.solvers.qp(obj, -q, G, h, A, b)['x']
        w = np.array(w)
    else:
        ones = np.matrix(np.ones(n))
        w = 1/delta*inv(Sigma)*(Mu+(delta-ones*inv(Sigma)* Mu.T)/(ones*inv(Sigma)* ones.T)*ones).T
    return w
    
def weights_bl(w_eq, Sigma, P, Q, Omega, delta=1.0, tau=0.05, short_selling=False):
    n = Sigma.shape[1]
    Sigma = np.asmatrix(Sigma)
    pi = delta * Sigma * np.asmatrix(w_eq)
    ts = tau * Sigma
    # Compute posterior estimate of the mean
    Mu_bar = inv(inv(ts)+np.asmatrix(P.T)*inv(Omega)*np.asmatrix(P)) \
            * (inv(ts)*pi + np.asmatrix(P.T)*inv(Omega)*np.asmatrix(Q).T)
    # Compute posterior estimate of the uncertainty in the mean
    posteriorSigma = inv(inv(ts)+np.asmatrix(P.T)*inv(Omega)*np.asmatrix(P))
    # Compute posterior weights based on uncertainty in mean
    if short_selling == False:
        obj = cvxopt.matrix(delta/2*posteriorSigma)
        q = cvxopt.matrix(Mu_bar, (n,1))
        A = cvxopt.matrix(1.0, (1,n))
        b = cvxopt.matrix(1.0)
        G = cvxopt.matrix(0.0, (n,n))
        G[::n+1] = -1.0
        h = cvxopt.matrix(0.0, (n,1))
        w = cvxopt.solvers.qp(obj, -q, G, h, A, b)['x']
        w = np.array(w)
    else:
        Sigma_bar = inv(Sigma + posteriorSigma)
        w = 1/delta*inv(Sigma_bar)*inv(posteriorSigma)\
                     *(inv(ts)*pi + np.asmatrix(P.T)*inv(Omega)*np.asmatrix(Q))
    return w

import numpy as np
from numpy import linalg 
from scipy.linalg import sqrtm

def torsion(Sigma, model, method='exact', max_niter=10000):
    n = Sigma.shape[0]    
    
    if model == 'pca':
        eigval, eigvec = linalg.eig(Sigma)
        idx = np.argsort(-eigval) 
        t = eigvec[:,idx]
        
    elif model == 'minimum-torsion':
        # C: correlation matrix
        sigma = np.sqrt(np.diag(Sigma))
        C = np.asmatrix(np.diag(1.0/sigma)) * np.asmatrix(Sigma) * np.asmatrix(np.diag(1.0/sigma))
        # Riccati root of correlation matrix
        c = sqrtm(C)
        if method == 'approximate':
            t = (np.asmatrix(sigma) / np.asmatrix(c)) * np.asmatrix(np.diag(1.0/sigma))
        elif method == 'exact':
            # initialize
            d = np.ones((n))
            f = np.zeros((max_niter))
            # iterating
            for i in range(max_niter):
                U = np.asmatrix(np.diag(d)) * c * c * np.asmatrix(np.diag(d))
                u = sqrtm(U)
                q = linalg.inv(u) * np.asmatrix(np.diag(d)) * c
                d = np.diag(q * c)
                pi = np.asmatrix(np.diag(d)) * q
                f[i] = linalg.norm(c - pi, 'fro')
                # if converge
                if i > 0 and abs(f[i]-f[i-1])/f[i] <= 1e-4:
                    f = f[0:i]
                    break
                elif i == max_niter and abs(f[max_niter]-f[max_niter-1])/f[max_niter] >= 1e-4:
                    print ('number of max iterations reached: n_iter = ' + str(max_niter))
            x = pi * linalg.inv(np.asmatrix(c))
            t = np.asmatrix(np.diag(sigma)) * x * np.asmatrix(np.diag(1.0/sigma))
    return t
    
    
def EffectiveBets(w, Sigma, t):
    w = np.asmatrix(w)
    p = np.asmatrix(np.asarray(linalg.inv(t.T) * w.T) * np.asarray(t * Sigma * w.T)) / (w * Sigma * w.T)  
    enb = np.exp(- p.T * np.log(p))
    return p, enb

# TODO edit this to remove file dependency
with open("SPY ticker.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content] 

data=pd.DataFrame()
for contents in content:
    data=pd.concat([data,yf.download(f"{contents}", start="2010-01-01", end="2020-01-01").iloc[:,4]],axis=1, sort=False)
data.columns = content

data1=data.dropna(how="all")
data1=data1.dropna(axis='columns',how="any")

returns = data1.pct_change().dropna()
ret = np.log(data1/data1.shift(1)).dropna()
data=data1



obs_window = 900
roll_window = 30
T = int((len(data) - obs_window)/30) +1 # number of windows
N = ret.shape[1] # number of stocks

p_pc = np.asmatrix(np.zeros((N,T)))
p_mt = np.asmatrix(np.zeros((N,T)))
enb_pc = np.zeros(T)
enb_mt = np.zeros(T)
margin_risk = np.asmatrix(np.zeros((N,T)))

for t in range(T):
    Sigma = np.asmatrix(np.cov(ret.ix[t*roll_window:t*roll_window+obs_window].T))
    w = weights_equal(Sigma)
    t_pc = torsion(Sigma, 'pca')
    t_mt = torsion(Sigma, 'minimum-torsion',  method='exact')
    p_pc[:,t], enb_pc[t] = EffectiveBets(w, Sigma, t_pc)
    p_mt[:,t], enb_mt[t] = EffectiveBets(w, Sigma, t_mt)
    margin_risk[:,t] = w.reshape((N,1)) * np.asarray(np.asmatrix(Sigma)*np.asmatrix(w).T) \
                       / (np.asmatrix(w) * np.asmatrix(Sigma) * np.asmatrix(w).T)
    



plotly.offline.plot([
    dict(z=np.sort(p_mt,axis=0)[::-1], type='surface'),
    dict(z=np.tile(np.asmatrix(w),(T,1)).T, showscale=False, opacity=0.9, type='surface')])

plotly.offline.plot([
    dict(z=np.sort(p_pc,axis=0)[::-1], type='surface'),
    dict(z=np.tile(np.asmatrix(w),(T,1)).T, showscale=False, opacity=0.9, type='surface')])
plotly.offline.plot([
    dict(z=np.sort(margin_risk,axis=0)[::-1], type='surface'),
    dict(z=np.tile(np.asmatrix(w),(T,1)).T, showscale=False, opacity=0.9, type='surface')])


obs_window = 900
roll_window = 30
T = int((len(data) - obs_window)/30) # number of windows
N = ret.shape[1] # number of stocks

p_pc = np.asmatrix(np.zeros((N,T)))
p_mt = np.asmatrix(np.zeros((N,T)))
enb_pc = np.zeros(T)
enb_mt = np.zeros(T)
margin_risk = np.asmatrix(np.zeros((N,T)))
wgt = np.asmatrix(np.zeros((N,T)))

for t in range(T):
    Sigma = np.asmatrix(np.cov(ret.ix[t*roll_window:t*roll_window+obs_window].T))
    Mu = np.mean(ret.ix[t*roll_window:t*roll_window+obs_window],axis=0)
    w = weights_mv(Mu, Sigma)
    wgt[:,t] = w
    w = w.T   
    t_pc = torsion(Sigma, 'pca')
    t_mt = torsion(Sigma, 'minimum-torsion',  method='exact')
    p_pc[:,t], enb_pc[t] = EffectiveBets(w, Sigma, t_pc)
    p_mt[:,t], enb_mt[t] = EffectiveBets(w, Sigma, t_mt)
    margin_risk[:,t] = w.reshape((N,1)) * np.asarray(np.asmatrix(Sigma)*np.asmatrix(w).T) \
                       / (np.asmatrix(w) * np.asmatrix(Sigma) * np.asmatrix(w).T)
    



plotly.offline.plot([
    dict(z=np.sort(p_mt,axis=0)[::-1], type='surface'),
    dict(z=np.sort(wgt,axis=0)[::-1], showscale=False, opacity=0.9, type='surface')])


plotly.offline.plot([
    dict(z=np.sort(p_pc,axis=0)[::-1], type='surface'),
    dict(z=np.sort(wgt,axis=0)[::-1], showscale=False, opacity=0.9, type='surface')])
    
plotly.offline.plot([
    dict(z=np.sort(margin_risk,axis=0)[::-1], type='surface'),
    dict(z=np.sort(wgt,axis=0)[::-1], showscale=False, opacity=0.9, type='surface')])
    
    
weight =pd.DataFrame(wgt)

import numpy as np 
import pandas as pd
import plotly
from plotly.graph_objs import Surface

obs_window = 900
roll_window = 30
T = int((len(data) - obs_window)/30) # number of windows
N = ret.shape[1] # number of stocks

p_pc = np.asmatrix(np.zeros((N,T)))
p_mt = np.asmatrix(np.zeros((N,T)))
enb_pc = np.zeros(T)
enb_mt = np.zeros(T)
margin_risk = np.asmatrix(np.zeros((N,T)))
wgt = np.asmatrix(np.zeros((N,T)))
delta = 1.0
tau = 0.05

for t in range(T):
    Sigma = np.asmatrix(np.cov(ret.ix[t*roll_window:t*roll_window+obs_window].T))
    Mu = np.mean(ret.ix[t*roll_window:t*roll_window+obs_window],axis=0)
    weq = weights_mv(Mu, Sigma)
    ts = tau * Sigma
    P = np.vstack((weq.reshape((429)),weights_equal(Sigma)))#do data1.shape[0] afterwards
    Q = np.array([np.sum(weq.reshape((429))*Mu)*(np.random.random()*2-1),\
              np.mean(Mu)*(np.random.random()*2-1)])
    Omega = np.dot(np.dot(P,ts),P.T) * np.eye(Q.shape[0])
    w = weights_bl(weq, Sigma, P, Q, Omega, delta = 1.0, tau = 0.05)
    wgt[:,t] = w
    w = w.T   
    t_pc = torsion(Sigma, 'pca')
    t_mt = torsion(Sigma, 'minimum-torsion',  method='exact')
    p_pc[:,t], enb_pc[t] = EffectiveBets(w, Sigma, t_pc)
    p_mt[:,t], enb_mt[t] = EffectiveBets(w, Sigma, t_mt)
    margin_risk[:,t] = w.reshape((N,1)) * np.asarray(np.asmatrix(Sigma)*np.asmatrix(w).T) \
                       / (np.asmatrix(w) * np.asmatrix(Sigma) * np.asmatrix(w).T)
    

import plotly
from plotly.graph_objs import Surface

plotly.offline.plot([
    dict(z=np.sort(p_mt,axis=0)[::-1], type='surface'),
    dict(z=np.sort(wgt,axis=0)[::-1], showscale=False, opacity=0.9, type='surface')])


plotly.offline.plot([
    dict(z=np.sort(p_pc,axis=0)[::-1], type='surface'),
    dict(z=np.sort(wgt,axis=0)[::-1], showscale=False, opacity=0.9, type='surface')])
    
plotly.offline.plot([
    dict(z=np.sort(margin_risk,axis=0)[::-1], type='surface'),
    dict(z=np.sort(wgt,axis=0)[::-1], showscale=False, opacity=0.9, type='surface')])
    
