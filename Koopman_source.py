### Load some packages 
### Import packages 
import networkx as nx # for handling graphs/networks 
import numpy as np # for basic scientific computing 
import pandas as pd # for basic scientific computing 
import matplotlib.pyplot as plt # for plotting
import matplotlib.gridspec as gridspec

import seaborn as sns

from scipy.special import binom
from scipy.linalg import subspace_angles

from scipy.integrate import solve_ivp
import Koopman_source as kp
from math import comb
#from copy import copy
import csv  
from scipy.optimize import minimize
from scipy.special import expit
from numpy import linalg as LA

def list2onehot(y, list_classes=None):
    """
    y = list of class lables of length n
    output = n x k array, i th row = one-hot encoding of y[i] (e.g., [0,0,1,0,0])
    """
    if list_classes is None:
        list_classes = list(np.sort(y))
    Y = np.zeros(shape = [len(y), len(list_classes)], dtype=int)
    for i in np.arange(Y.shape[0]):
        for j in np.arange(len(list_classes)):
            if y[i] == list_classes[j]:
                Y[i,j] = 1
    return Y


def SBM(W, c):
    # Stochastic block model; 
    # W = (k x k) community weight matrix 
    # c = (n x 1), entries from [k]; community assignment vector
    k = W.shape[0]
    n = len(c)
    
    # C = (n x k) one-hot encoding of community assignment matrix
    C = list2onehot(c, list_classes=[i for i in range(k)])

    # C = (n x n) probability matrix = expected adjacency matrix = C W C.T
    P = C @ W @ C.T
    
    # Now sample the edges according to P
    G = weight2graph(P)
    return G


def weight2digraph(P):
    # P = (n x n) (asymmetric) weight (= edge probability) matrix 
    n = P.shape[0]
    
    # Sample the edges according to P
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    nodes = list(G.nodes())

    for i in np.arange(n):
        for j in np.arange(n):
            U = np.random.rand()
            if U < P[i,j]:
                G.add_edge(nodes[i],nodes[j])
    return G


def weight2graph(P):
    # P = (n x n) symmetric weight (= edge probability) matrix 
    n = P.shape[0]
    
    # Sample the edges according to P
    G = nx.Graph()
    G.add_nodes_from(range(n))
    nodes = list(G.nodes())

    for i in np.arange(n):
        for j in np.arange(i+1,n):
            U = np.random.rand()
            if U < P[i,j]:
                G.add_edge(nodes[i],nodes[j])
    return G


def RW(G, x0=None, steps=1, return_history=False):
    # simple symmetric random walk on graph G 
    # initialization at x0
    if x0 is None:
        x = np.random.choice(G.nodes())
    else:
        x = x0
    
    history = []
    for i in np.arange(steps):
        if len(list(G.neighbors(x))) == 0:
            print("RW is stuck at isolated node")
            x = np.random.choice(G.nodes()) # re-initialize uniformly at random
        else: 
            x = np.random.choice(list(G.neighbors(x)))

        if return_history:
            history.append(x)
        
    if not return_history: 
        return x 
    else: 
        return history


# Source library for the Koopman theory research.
# Discrete dynamical system

def toy_d1(t, x0, mu0, lambda0): # Here, d1 stands for the first discrete example.
    # t must be a positive integer.
    # x0 is the initial value x(0) = x0
    #mu0, lambda0 = kappa[0], kappa[1]
    x_mat = np.zeros([len(x0),t]) # matrix for saving trajectories. It does not save the initial value.
    x_old = x0.copy()
    x_new = x_old
    for i in range(t):
        x_new[0] = (1+mu0)*x_old[0]
        x_new[1] = (1+lambda0)*x_old[1]
        x_mat[:,i] = x_new
    return x_mat

def Lorenz(x_old, sigma0, rho0, beta0, dt):
    # dt = 0.1
    x_new = x_old.copy()
    x_new[0] = x_old[0] + sigma0 * (x_old[1]-x_old[0])*dt
    x_new[1] = x_old[1] + (x_old[0]*(rho0-x_old[2]) - x_old[2])*dt
    x_new[2] = x_old[2] + (x_old[0]*x_old[2] - beta0*x_old[2])*dt
    return x_new

def Lotka_Voltera(x_old, rho0, sigma0, alpha0, beta0, dt):
    # dt = 1
    x_new = x_old.copy()
    x_new[0] = rho0 * x_old[0] * (1- x_old[0]) - alpha0 * x_old[0] * x_old[1]
    x_new[1] = sigma0 * x_old[1] + beta0 * x_old[0] * x_old[1]
    return x_new

def Duffing(x_old, alpha0, beta0, delta0, dt):
    x_new = x_old.copy()
    x_new[0] = x_old[0] + x_old[1]*dt
    x_new[1] = x_old[1] + (-delta0*x_old[1] - alpha0 * x_old[0] - beta0 * np.power(x_old[0],3))*dt
    return x_new

def psi_id(x):
    return x
    
def psi_d1(x):
    return x

def psi_d2(x, max_order):
    d = 2 # the number of variables or the dimension of a dynamical system.
    num_obs = comb(max_order+d, d)# (d+max_order)Cd = dH0 + dH1 + \cdots dHmax_order by the hockey stick rule where H is repeated combination nHr = (n+r-1)Cr
    obs = np.zeros(num_obs) # the total number of observable functions, psi_m's.
    idx = 0 
    ind_mat =[]
    for i in range(max_order+1):
        for j in range(i+1):
            obs[idx] = np.power(x[0], i-j) * np.power(x[1], j)
            idx += 1
            ind_mat.append([i-j,j])
    return obs, ind_mat

def Laguerre_poly(x, n):
    if n==0:
        return 1
    if n==1:
        return -x+1
    if n>1:
        return ((2*n-1-x)*Laguerre_poly(x, n-1) - (n-1)*Laguerre_poly(x, n-2))/n
    # if n==2:
    #     return 1/2 * (x**2 - 4*x + 2)
    # if n==3:
    #     return 1/6 * (-x**3 + 9*x**2 - 18*x + 6)
    # if n==4:
    #     return 1/24 * (x**4 - 16*x**3 + 72*x**2 - 96*x + 24)
    # print("For now, n must be less than 5!")
    # return None

def psi_d2_Laguerre(x):
    d = 2 # the number of variables or the dimension of a dynamical system.
    max_order = 2
    num_obs = comb(max_order+d, d)# (d+max_order)Cd = dH0 + dH1 + \cdots dHmax_order by the hockey stick rule where H is repeated combination nHr = (n+r-1)Cr
    obs = np.zeros(num_obs) # the total number of observable functions, psi_m's.
    idx = 0 
    for i in range(max_order+1):
        for j in range(i+1):
            obs[idx] = Laguerre_poly(x[0], i-j) * Laguerre_poly(x[1], j)
            idx += 1
    return obs

def psi_radial(x, cj, gamma):
    return np.exp(-gamma *LA.norm(x-cj, 2))

def psi_Laguerre(x, d, max_order):
    # d: the number of variables or the dimension of a dynamical system.
    num_obs = comb(max_order+d, d)# (d+max_order)Cd = dH0 + dH1 + \cdots dHmax_order by the hockey stick rule where H is repeated combination nHr = (n+r-1)Cr
    obs = [] #np.zeros(num_obs) # the total number of observable functions, psi_m's.
    idx = 0 
    ind_mat =[]
    inc_ord = True
    if d == 3:
        if inc_ord == True:
            for i in range(max_order+1):
                for j in range(i+1):
                    for k in range(j+1):
                        obs.append(Laguerre_poly(x[0], i-j) * Laguerre_poly(x[1], j-k) * Laguerre_poly(x[2], k))
                        ind_mat.append([i-j,j-k,k])
                        # idx += 1
        else:
            for i in range(max_order+1):
                for j in range(max_order-i+1):
                    for k in range(max_order-i-j+1):
                        obs.append(Laguerre_poly(x[0], i) * Laguerre_poly(x[1], j) * Laguerre_poly(x[2], k))
                        ind_mat.append([i,j,k])
    if d == 2:
        if inc_ord == True:
            for i in range(max_order+1):
                for j in range(i+1):
                    obs.append(Laguerre_poly(x[0], i-j) * Laguerre_poly(x[1], j))
                    ind_mat.append([i-j,j])
        else:
            for i in range(max_order+1):
                for j in range(max_order-i+1):
                    obs.append(Laguerre_poly(x[0], i) * Laguerre_poly(x[1], j))
                    ind_mat.append([i,j])
    
    return np.array(obs), ind_mat


### ====== Page-Rank Algorithm ======================
def pagerank(M, num_iterations=100, d=0.85):
    N = M.shape[1]
    
    # Initialize the PageRank vector
    v = np.ones(N) / N
    
    # Compute the column sum
    out_degree = np.sum(M, axis=0)
    
    # Normalize the adjacency matrix
    M_hat = M / out_degree
    
    # PageRank iteration
    for _ in range(num_iterations):
        v_next = (1 - d) / N + d * M_hat.dot(v)
        if np.allclose(v, v_next):
            return v_next
        v = v_next
    
    return v

def subsampled_pagerank(M, num_iterations=100, d=0.85, subsample_rate=0.1, epsilon=1e-8):
    N = M.shape[0]
    
    # Normalize the adjacency matrix
    M_hat = M / np.sum(M, axis=0)
    
    # Initialize the PageRank vector
    v = np.ones(N) / N
    
    for _ in range(num_iterations):
        v_prev = v.copy()
        
        # Subsampling
        mask = np.random.random(N) < subsample_rate
        subsampled_M = M_hat[:, mask]
        subsampled_v = v_prev[mask]
        
        # PageRank update
        v = (1 - d) / N + d * subsampled_M.dot(subsampled_v / subsample_rate)
        
        # Check for convergence
        if np.sum(np.abs(v - v_prev)) < epsilon:
            break
    
    return v / np.sum(v)
