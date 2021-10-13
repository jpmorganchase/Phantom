import datetime
import logging
import typing
from dataclasses import dataclass

import gym
import mercury as me
import numpy as np
import phantom as ph

LOG_LEVEL = "WARN"

#######################################
##  Payloads
#######################################
# These are the content of the messages
# that will be sent between agents


@dataclass(frozen=True)
class ImpressionRequest(me.Payload):
    """
    The message indicating that a user  is visiting a website and might be
    interested in an advertisement offer

    Attributes:
    -----------
    timestamp (int):  the time of the impression
    user_id (int):    the unique and anonymous identifier of the user

    Methods:
    --------
    generate_random():  helper method to generate random impressions

    """

    timestamp: int
    user_id: int

    @classmethod
    def generate_random(cls):
        return cls(
            timestamp=datetime.datetime.now().timestamp(),
            user_id=np.random.choice([1, 2]),
        )


@dataclass(frozen=True)
class Bid(me.Payload):
    """
    The message sent by the advertiser to the exchange
    to win the impression

    Attributes:
    -----------
    bid (float):    the cost charged to the advertiser
    theme (str):    the theme of the ad that will be displayed
    user_id(str):   the user identifier

    """

    bid: float
    theme: str
    user_id: int


@dataclass(frozen=True)
class AuctionResult(me.Payload):
    """
    The message sent by the exchange to the advertiser
    to inform her of the auction's result

    Attributes:
    -----------
    cost (float):           the cost charged to the advertiser
    winning_bid (float):    the highest bid during this auction

    """

    cost: float
    winning_bid: float


@dataclass(frozen=True)
class Ads(me.Payload):
    """
    The message sent by an advertisers containing the ads to show to the user.
    For simplicity, it only contains a theme.

    Attributes:
    -----------
    advertiser_id (str):    the theme of the ads
    theme (str):            the theme of the ads
    user_id (int):          the user_id that will receive the ads

    """

    advertiser_id: str
    theme: str
    user_id: int


@dataclass(frozen=True)
class ImpressionResult(me.Payload):
    """
    The result of the ad display. i.e whether or not the user clicked
    on the ad

    Attributes:
    -----------
    clicked (bool):    whether or not the user clicked on the ad

    """

    clicked: bool


#######################################
##  Agents
#######################################
# Definitions of our 3 types of agents:
#   - Publisher
#   - Advertiser
#   - AdExchange


class PublisherAgent(ph.ZeroIntelligenceAgent):
    """
    A `PublisherAgent` generates `ImpressionRequest` which corresponds to
    real-estate on their website rented to advertisers to display their ads.

    Attributes:
    -----------
    _USER_CLICK_PROBABILITIES (dict): helper dictionary containing the probability
        to click on the ads for each user. For simplicity, we hardcode these values,
        however more a advanced logic could also be implemented
    """

    _USER_CLICK_PROBABILITIES = {
        1: {"sport": 0.1, "travel": 0.8, "science": 0.2, "tech": 0.7},
        2: {"sport": 0.7, "travel": 0.3, "science": 0.8, "tech": 0.1},
    }

    def __init__(self, agent_id: str, exchange_id: str):
        super().__init__(agent_id)

        self.exchange_id = exchange_id

    def decode_action(self, _ctx: me.Network.Context, _action: np.ndarray):
        """@override
        Method called at each step of the episode and generating a new impression
        each time.
        """
        return ph.packet.Packet(
            messages={self.exchange_id: [ImpressionRequest.generate_random()]}
        )

    @me.actors.handler(Ads)
    def handle_ads(self, _ctx: me.Network.Context, msg: me.Message):
        """
        Method to process messages with the payload type `Ads`

        Note:
        -----
        We register the type of payload to process via the `me.handle` decorator

        Params:
        -------
        ctx (me.Network.Context): the partially observable context available for
            the agent
        msg (me.Message): the message received by the agent.

        Returns:
        --------
        receiver_id (me.ID): the unique identifier of the agent the messages are
            intended to
        messages ([me.Message]): the messages to send

        """
        _get_logger().debug("PublisherAgent %s ads: %s", self.id, msg.payload)

        clicked = np.random.binomial(
            1, self._USER_CLICK_PROBABILITIES[msg.payload.user_id][msg.payload.theme]
        )
        yield (msg.payload.advertiser_id, [ImpressionResult(clicked=clicked)])


class AdvertiserAgent(ph.Agent):
    """
    An `AdvertiserAgent` learns to bid efficiently and within its budget limit, on an impression
    in order to maximize the number of clicks it gets.
    For this implementation an advertiser is associated with a `theme` which will impact the
    probability of a user to click on the ad.

    Observation Space:
        - budget left
        - user id
        - user age
        - user zipcode

    Action Space:
        - bid amount
    """

    def __init__(
        self, agent_id: str, exchange_id: str, budget: float, theme: str = "generic"
    ):
        self.exchange_id = exchange_id
        self.budget = budget
        self.theme = theme

        self.left = budget

        super().__init__(agent_id)

    def pre_resolution(self, _ctx: me.Network.Context):
        """@override
        The `pre_resolution` method is called at the beginning of each step.
        We use this method to reset the number of clicks received during the step.
        """
        self.step_clicks = 0

    @me.actors.handler(ImpressionRequest)
    def handle_impression_request(self, ctx: me.Network.Context, msg: me.Message):
        """
        Once an `ImpressionRequest` is received we cache the information about the user.

        Note:
        -----
        We receive the user id in the message but we collect extra user information from
        the `ctx` object.
        """
        _get_logger().debug(
            "AdvertiserAgent %s impression request: %s", self.id, msg.payload
        )

        self._current_user_id = msg.payload.user_id

        self._current_age = ctx[self.exchange_id].users_info[self._current_user_id][
            "age"
        ]
        self._current_zipcode = ctx[self.exchange_id].users_info[self._current_user_id][
            "zipcode"
        ]

        yield from ()

    @me.actors.handler(AuctionResult)
    def handle_auction_result(self, _ctx: me.Network.Context, msg: me.Message):
        """
        If the `AdvertiserAgent` wins the auction it needs to update its budget left.
        """
        _get_logger().debug(
            "AdvertiserAgent %s auction result: %s", self.id, msg.payload
        )

        self.left -= msg.payload.cost

        yield from ()

    @me.actors.handler(ImpressionResult)
    def handle_impression_result(self, _ctx: me.Network.Context, msg: me.Message):
        """
        When the result of the ad display is received, update the number of clicks.
        """
        _get_logger().debug(
            "AdvertiserAgent %s impression result: %s", self.id, msg.payload
        )

        self.step_clicks += int(msg.payload.clicked)
        self.total_clicks += int(msg.payload.clicked)

        yield from ()

    def encode_obs(self, _ctx: me.Network.Context):
        """@override
        The observation will help learn the policy.

        For this use case we pass:
            - the budget the agent has left
            - the user id the impression will be for
            - the user's age
            - the user's zipcode
        """
        return np.array(
            [self.left, self._current_user_id, self._current_age, self._current_zipcode]
        )

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        """@override
        We receive the "optimal" bid from the learnt Policy and send a message to the
        exchange to try to win the impression.
        """
        _get_logger().debug("AdvertiserAgent %s decode action: %s", self.id, action)
        msgs = []

        self.bid = min(action[0], self.left)

        if self.bid > 0.0:
            msg = Bid(bid=self.bid, theme=self.theme, user_id=self._current_user_id)
            msgs.append(msg)
        return ph.packet.Packet(messages={self.exchange_id: msgs})

    def compute_reward(self, _ctx: me.Network.Context) -> float:
        """@override
        The goal is to maximize the number of clicks so the per-step reward
        is the number of clicks received at the current timestep.
        """
        return self.step_clicks

    def is_done(self, _ctx: me.Network.Context) -> bool:
        """@override
        This agent cannot perform any more bids if its budget is 0.
        """
        return self.left <= 0

    def reset(self):
        """@override
        Reset method called before each episode to clear the state of the agent.
        """
        self.left = self.budget

        self.step_clicks = 0
        self.total_clicks = 0

        self.bid = 0.0
        self._current_user_id = 0.0
        self._current_age = 0.0
        self._current_zipcode = 0.0

    def get_observation_space(self):
        """@override
        `gym.spaces` defintion of the Observation space
        """
        return gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([self.budget, 5.0, np.inf, np.inf]),
        )

    def get_action_space(self):
        """@override
        `gym.spaces` defintion of the Action space
        """
        return gym.spaces.Box(low=np.array([0.0]), high=np.array([self.budget]))


class AdExchangeAgent(me.actors.SimpleSyncActor):
    """
    The `AdExchangeAgent` is actually just an actor who reacts to messages reveived.
    It doesn't perform any action on its own.
    """

    @dataclass(frozen=True)
    class AdExchangeView(me.actors.View):
        """
        The view is used to expose additional information to other actors in the system.
        It is accessible via the `me.Network.Context` object passed as a parameters
        in the appropriate methods.

        For this use case we want to expose users information to the advertisers to help them
        decide on their bid
        """

        users_info: dict

    def __init__(
        self,
        agent_id: str,
        publisher_id: str,
        advertiser_ids: typing.Iterable = tuple(),
        strategy: str = "second",
    ):
        super().__init__(agent_id)

        self.publisher_id = publisher_id
        self.advertiser_ids = advertiser_ids

        self.strategy = strategy

    def view(self, neighbour_id=None) -> me.actors.View:
        """@override
        Method to provide extra information about the user. This information
        is made available only for advertisers in a pull fashion, i.e the
        advertiser needs to access the information explicitely via the `ctx`
        object if it wants to use it.
        """
        if neighbour_id.startswith("ADV"):
            return self.AdExchangeView(
                actor_id=self._id,
                users_info={
                    1: {"age": 18, "zipcode": 94025},
                    2: {"age": 40, "zipcode": 90250},
                },
            )
        else:
            return super().view(neighbour_id)

    @me.actors.handler(ImpressionRequest)
    def handle_impression_request(self, _ctx: me.Network.Context, msg: me.Message):
        """
        The exchange acts as an intermediary between the publisher and the
        advertisers, upon the reception of an `ImpressionRequest`, the exchange
        simply forward that request to the advertisers
        """
        _get_logger().debug("AdExchange impression request %s", msg)

        for adv_id in self.advertiser_ids:
            yield (adv_id, [msg.payload])

    def handle_batch(self, ctx: "me.Network.Context", batch: me.message.Batch):
        """@override
        We override the method `handle_batch` to consume all the bids messages
        as one block in order to perform the auction. The batch object contains
        all the messages that were sent to the actor.

        Note:
        -----
        The default logic is to consume each message individually.
        """
        bids = []
        for sender_id in batch:
            for message in batch.messages_from(sender_id):
                if isinstance(message.payload, Bid):
                    bids.append(message)
                else:
                    yield from self.handle_message(ctx, message)

        if bids:
            yield from self.auction(bids)

    def auction(self, bids):
        """
        Classic auction mechanism. We implement two types of auctions here:
            - first price: the cost corresponds to the highest bid
            - second price: the cost corresponds to the second highest bid
        In both cases the highest bid wins.
        """
        if self.strategy == "first":
            winner, cost = self._first_price_auction(bids)
        elif self.strategy == "second":
            winner, cost = self._second_price_auction(bids)
        else:
            raise ValueError(f"Unknown auction strategy: {self.strategy}")

        _get_logger().debug("AdExchange auction done winner: %s cost: %s", winner, cost)

        yield (
            self.publisher_id,
            [
                Ads(
                    advertiser_id=winner.sender_id,
                    theme=winner.payload.theme,
                    user_id=winner.payload.user_id,
                )
            ],
        )

        for adv_id in self.advertiser_ids:
            adv_cost = cost if adv_id == winner.sender_id else 0.0
            yield (
                adv_id,
                [AuctionResult(cost=adv_cost, winning_bid=winner.payload.bid)],
            )

    def _first_price_auction(self, bids):
        sorted_bids = sorted(bids, key=lambda m: m.payload.bid, reverse=True)
        winner = sorted_bids[0]
        cost = sorted_bids[0].payload.bid
        return winner, cost

    def _second_price_auction(self, bids):
        sorted_bids = sorted(bids, key=lambda m: m.payload.bid, reverse=True)
        winner = sorted_bids[0]
        cost = (
            sorted_bids[1].payload.bid if len(bids) > 1 else sorted_bids[0].payload.bid
        )
        return winner, cost


#######################################
##  Environment
#######################################


class DigitalAdsEnv(ph.PhantomEnv):

    env_name: str = "digital-ads-v1"

    def __init__(self, seed: int = 0):
        # agent ids
        self.exchange_id = "ADX"
        self.publisher_id = "PUB"

        # The `PublisherAgent`
        publisher_agent = PublisherAgent(
            self.publisher_id, exchange_id=self.exchange_id
        )

        # The learning `AdvertiserAgent`s
        advertiser_agents = [
            AdvertiserAgent("ADV_1", self.exchange_id, budget=20.0, theme="travel"),
            AdvertiserAgent("ADV_2", self.exchange_id, budget=10.0, theme="travel"),
            AdvertiserAgent("ADV_3", self.exchange_id, budget=20.0, theme="tech"),
            AdvertiserAgent("ADV_4", self.exchange_id, budget=30.0, theme="tech"),
            AdvertiserAgent("ADV_5", self.exchange_id, budget=10.0, theme="sport"),
            AdvertiserAgent("ADV_6", self.exchange_id, budget=5.0, theme="science"),
            AdvertiserAgent("ADV_7", self.exchange_id, budget=50.0, theme="science"),
            AdvertiserAgent("ADV_8", self.exchange_id, budget=20.0, theme="sport"),
        ]
        self.advertiser_ids = [a.id for a in advertiser_agents]

        # The `AdExchangeAgent` that makes the intermediary between publishers and advertisers
        exchange_agent = AdExchangeAgent(
            self.exchange_id,
            publisher_id=self.publisher_id,
            advertiser_ids=self.advertiser_ids,
        )

        # Building the network defining all the actors and connecting them
        actors = [exchange_agent, publisher_agent] + advertiser_agents
        network = me.Network(me.resolvers.UnorderedResolver(chain_limit=5), actors)
        network.add_connections_between([self.exchange_id], [self.publisher_id])
        network.add_connections_between([self.exchange_id], self.advertiser_ids)
        network.add_connections_between([self.publisher_id], self.advertiser_ids)

        super().__init__(network=network, n_steps=100, seed=seed)

    def reset(self):
        """@override
        For the first step of the episode we only want the `PublisherAgent` to
        send an `ImpressionRequest`, because there is no request the
        `AdvertiserAgent`s should not send any messages. To do so,
        we only return the observations for the `PublisherAgent` from the
        `reset` method. This has the ben
        """
        obs = super().reset()

        return {aid: o for aid, o in obs.items() if aid in [self.publisher_id]}

    def is_done(self):
        """@override
        The episode is done when all the advertiser agents are dones or
        when the number of steps has reached its maximum
        """
        return self.clock.is_terminal or len(self._dones) == len(self.advertiser_ids)


#######################################
##  Metrics
#######################################


class AdvertiserTotalClicks(ph.logging.Metric[float]):
    def __init__(self, agent_id: str) -> None:
        self.agent_id: str = agent_id

    def extract(self, env: ph.PhantomEnv) -> float:
        """@override
        Extracts the per-step value to track
        """
        return env[self.agent_id].total_clicks


class AdvertiserAverageBidUser(ph.logging.Metric[float]):
    def __init__(self, agent_id: str, user_id: int) -> None:
        self.agent_id: str = agent_id
        self.user_id: int = user_id

    def extract(self, env: ph.PhantomEnv) -> float:
        """@override
        Extracts the per-step value to track
        """
        if env[self.agent_id]._current_user_id == self.user_id:
            return env[self.agent_id].bid
        return np.nan

    def reduce(self, values) -> float:
        """@override
        The default logic returns the last step value,
        here we are interested in the average bid value
        """
        return np.nanmean(values)


METRICS = {}
for aid in (f"ADV_{i}" for i in range(1, 9)):
    METRICS[f"{aid}/clicks"] = AdvertiserTotalClicks(aid)
    METRICS[f"{aid}/avg_bid_user_1"] = AdvertiserAverageBidUser(aid, 1)
    METRICS[f"{aid}/avg_bid_user_2"] = AdvertiserAverageBidUser(aid, 2)

#######################################
##  Params
#######################################

ph.train(
    experiment_name="digital-ads",
    algorithm="PPO",
    num_workers=0,
    num_episodes=5,
    env=DigitalAdsEnv,
    env_config={"seed": 0},
    metrics=METRICS,
)

#######################################
##  Helpers
#######################################


def _get_logger(log_level=LOG_LEVEL):
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("digital-ads")
    logger.setLevel(log_level)
    return logger
