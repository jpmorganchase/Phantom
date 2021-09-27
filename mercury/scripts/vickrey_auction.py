import dataclasses

import numpy as np
import mercury

from mercury import Network, Batch
from mercury.actors import Actor, SimpleSyncActor, handler
from mercury.resolvers import UnorderedResolver


@dataclasses.dataclass(frozen=True)
class PlaceBid(mercury.Payload):
    price: float


@dataclasses.dataclass(frozen=True)
class Allocation(mercury.Payload):
    cash: float
    quantity: int


class Auctioneer(Actor):
    def __init__(self, actor_id: mercury.ID) -> None:
        Actor.__init__(self, actor_id)

    def handle_batch(self, ctx, batch):
        bids = [(nid, min(batch[nid], key=lambda p: p.price)) for nid in batch]
        bids.sort(key=lambda x: x[1].price)

        yield bids[-1][0], [Allocation(cash=-bids[-2][1].price, quantity=1)]


class Bidder(SimpleSyncActor):
    def __init__(self, aid: str, cash: float, inventory: int) -> None:
        super().__init__(aid)

        self.cash = cash
        self.inventory = inventory

    @handler(Allocation)
    def handle_allocation(self, ctx, message):
        self.cash += message.payload.cash
        self.inventory += message.payload.quantity

        yield from ()


class VickreyAuction(Network):
    def __init__(self, n_bidders: int) -> None:
        Network.__init__(self, UnorderedResolver(2), actors=[Auctioneer("auctioneer")])

        for i in range(n_bidders):
            bidder_id = "bidder_{}".format(i)

            self.add_actor(Bidder(bidder_id, 0.0, 0))
            self.add_connection(bidder_id, "auctioneer")

    def send_bids(self, bid_map):
        self.send_to(
            "auctioneer",
            {bidder_id: [PlaceBid(bid)] for bidder_id, bid in bid_map.items()},
        )

        return self


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("vickrey_sim")
    parser.add_argument("n_rounds", type=int)
    parser.add_argument("--n_bidders", type=int, default=5)

    args = parser.parse_args()
    game = VickreyAuction(args.n_bidders)

    for _ in range(args.n_rounds):
        game.send_bids(
            {
                "bidder_{}".format(i): np.random.normal(loc=i, scale=2.0)
                for i in range(args.n_bidders)
            }
        ).resolve()

    for i in range(args.n_bidders):
        bidder = game["bidder_{}".format(i)]
        avg_price = bidder.cash / bidder.inventory if bidder.inventory else -np.inf
        market_share = bidder.inventory / args.n_rounds

        print("{}:".format(bidder.id), avg_price, "\t({:.0%})".format(market_share))
