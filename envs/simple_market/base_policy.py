import numpy as np


def BuyerPolicy(obs):
    # obs: min_price, demand, value
    if obs[1] and (obs[0] <= obs[2]):
        action = obs[1]  # buy
    else:
        action = 0  # no-buy
    return action


def SellerPolicy(obs):
    # random price in [0,1]
    action = np.random.uniform()
    return action
