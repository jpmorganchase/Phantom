# Phantom Environments

This directory contains a collection of sample environments for the Phantom framework.


## Simple Market

A simple market environment with buyers whose utility depends on their private value for
the good and sellers with infinite supply and no inventory cost. The environment has a
second version - a "leaky" simple market environment where a "leaky buyer" shares the
price of a targeted "victim seller" with an adversarial seller.

## Supply Chain

This example was created as part of the tutorial in the Phantom documentation. Please
see the tutorial for a full description of this environment.

### `supply-chain-1`

A basic environment modelling the transport of stock between a factory, a shop and
multiple customers. The goal is for the shop to learn the optmimum amount of stock to
hold at any one time.

### `supply-chain-2`

This builds on the initial supply chain example, making use of more features of Phantom
such as shared policies, metrics and supertypes.


## Digital Ads Market

An environment simulating the Digital Ads Market where a single publisher sells
impressions to eight advertisers via an ad exchange. The goal is for the publisher to
learn the optimal bidding strategy maximizing the number of clicks they get on their ads.
