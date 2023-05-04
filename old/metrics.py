from statsmodels.tsa.stattools import adfuller as adfuller
import numpy as np
import pandas as pd

# TODO: Turn these into classes that extend a Metrics class? Only makes sense if they have shared methods and attributes

def gini(x):
    """
    Gini coefficient measures the inequality in the pool.

    @Params
        x : Array 
            Token weights or balances.
    
    @Returns
        coef : Double 
            Gini coefficient 
    """
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    coef = total / (len(x)**2 * np.mean(x))
    # print(f"The Gini coefficient of the pool is {coef=:.2f}")
    return coef

def shannons_entropy(x):
    """
    Imagine a pool is a basket and each unit of each asset is a ball with that asset's color.
    Shannon entropy [loosely] measures how easy it is to predict the color of a ball picked at random.

    @Params
        x : Array
            Token weights or balances
    
    @Returns
        entropy : Double
            Shannon's Entropy measurement
    """
    proportions = x / np.sum(x)
    entropy = -np.sum(proportions * np.log2(proportions))
    # print(f"The entropy of the pool is {entropy:.2f}")
    return entropy

def swap_flow_ma(x, window):
    