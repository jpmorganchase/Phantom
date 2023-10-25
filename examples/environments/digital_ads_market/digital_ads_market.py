import datetime
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import cloudpickle
import gymnasium as gym
import numpy as np
import phantom as ph

LOG_LEVEL = "WARN"

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("digital-ads")
logger.setLevel(LOG_LEVEL)


#######################################
##  Payloads
#######################################
# These are the content of the messages
# that will be sent between agents


@ph.msg_payload()
class ImpressionRequest:
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

    timestamp: float
    user_id: int

    @classmethod
    def generate_random(cls):
        return cls(
            timestamp=datetime.datetime.now().timestamp(),
            user_id=np.random.choice([1, 2]),
        )


@ph.msg_payload()
class Bid:
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


@ph.msg_payload()
class AuctionResult:
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


@ph.msg_payload()
class Ads:
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


@ph.msg_payload()
class ImpressionResult:
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


class PublisherPolicy(ph.Policy):
    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0])


class PublisherAgent(ph.Agent):
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
        1: {"sport": 0.0, "travel": 1.0, "science": 0.2, "tech": 0.8},
        2: {"sport": 1.0, "travel": 0.0, "science": 0.7, "tech": 0.1},
    }

    def __init__(self, agent_id: str, exchange_id: str, user_click_proba: dict = None):
        super().__init__(agent_id)

        self.exchange_id = exchange_id
        self.user_click_proba = user_click_proba or self._USER_CLICK_PROBABILITIES

    def generate_messages(self, ctx: ph.Context):
        return [(self.exchange_id, ImpressionRequest.generate_random())]

    @ph.agents.msg_handler(Ads)
    def handle_ads(self, _ctx: ph.Context, msg: ph.Message):
        """
        Method to process messages with the payload type `Ads`

        Note:
        -----
        We register the type of payload to process via the `ph.agents.handle_msg` decorator

        Params:
        -------
        ctx (ph.Context): the partially observable context available for
            the agent
        msg (ph.MsgPayload): the message received by the agent.

        Returns:
        --------
        receiver_id (ph.AgentID): the unique identifier of the agent the messages are
            intended to
        messages ([ph.MsgPayload]): the messages to send

        """
        logger.debug("PublisherAgent %s ads: %s", self.id, msg.payload)

        clicked = np.random.binomial(
            1, self.user_click_proba[msg.payload.user_id][msg.payload.theme]
        )
        return [(msg.payload.advertiser_id, ImpressionResult(clicked=clicked))]


class AdvertiserAgent(ph.StrategicAgent):
    """
    An `AdvertiserAgent` learns to bid efficiently and within its budget limit, on an impression
    in order to maximize the number of clicks it gets.
    For this implementation an advertiser is associated with a `theme` which will impact the
    probability of a user to click on the ad.

    Observation Space:
        - budget left
        - user id
        # - user age
        # - user zipcode

    Action Space:
        - bid amount
    """

    @dataclass
    class Supertype(ph.Supertype):
        # The overall budget to spend during the episode
        budget: float

    def __init__(self, agent_id: str, exchange_id: str, theme: str = "generic"):
        self.exchange_id = exchange_id
        self.theme = theme

        self.action_space = gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]))

        super().__init__(agent_id)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "type": self.type.to_obs_space(),
                "budget_left": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float64
                ),
                "user_id": gym.spaces.Discrete(2),
                # "user_age": gym.spaces.Box(low=0.0, high=100., shape=(1,), dtype=np.float64),
                # "user_zipcode": gym.spaces.Box(low=0.0, high=99999., shape=(1,), dtype=np.float64),
            }
        )

    def pre_message_resolution(self, _ctx: ph.Context):
        """@override
        The `pre_resolution` method is called at the beginning of each step.
        We use this method to reset the number of clicks received during the step.
        """
        self.step_clicks = 0
        self.step_wins = 0

    @ph.agents.msg_handler(ImpressionRequest)
    def handle_impression_request(self, ctx: ph.Context, msg: ph.Message):
        """
        Once an `ImpressionRequest` is received we cache the information about the user.

        Note:
        -----
        We receive the user id in the message but we collect extra user information from
        the `ctx` object.
        """
        logger.debug("AdvertiserAgent %s impression request: %s", self.id, msg.payload)

        self._current_user_id = msg.payload.user_id

        self._current_age = ctx[self.exchange_id].users_info[self._current_user_id][
            "age"
        ]
        self._current_zipcode = ctx[self.exchange_id].users_info[self._current_user_id][
            "zipcode"
        ]

        self.total_requests[self._current_user_id] += 1

    @ph.agents.msg_handler(AuctionResult)
    def handle_auction_result(self, _ctx: ph.Context, msg: ph.MsgPayload):
        """
        If the `AdvertiserAgent` wins the auction it needs to update its budget left.
        """
        logger.debug("AdvertiserAgent %s auction result: %s", self.id, msg.payload)

        self.step_wins += int(msg.payload.cost != 0.0)
        self.total_wins[self._current_user_id] += int(msg.payload.cost != 0.0)
        self.left -= msg.payload.cost

    @ph.agents.msg_handler(ImpressionResult)
    def handle_impression_result(self, _ctx: ph.Context, msg: ph.MsgPayload):
        """
        When the result of the ad display is received, update the number of clicks.
        """
        logger.debug("AdvertiserAgent %s impression result: %s", self.id, msg.payload)

        self.step_clicks += int(msg.payload.clicked)
        self.total_clicks[self._current_user_id] += int(msg.payload.clicked)

    def encode_observation(self, _ctx: ph.Context):
        """@override
        The observation will help learn the policy.

        For this use case we pass:
            - the budget the agent has left
            - the user id the impression will be for
            - the user's age
            - the user's zipcode
        """
        if self._current_user_id != 0:
            return {
                "type": self.type.to_obs_space_compatible_type(),
                "budget_left": np.array(
                    [self.left / self.type.budget], dtype=np.float64
                ),
                "user_id": self._current_user_id - 1,
                # "user_age": np.array([self._current_age / 100.], dtype=np.float64),
                # "user_zipcode": np.array([self._current_zipcode / 99999.], dtype=np.float64),
            }

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        """@override
        We receive the "optimal" bid from the learnt Policy and send a message to the
        exchange to try to win the impression.
        """
        logger.debug("AdvertiserAgent %s decode action: %s", self.id, action)
        msgs = []

        self.bid = min(action[0] * self.type.budget, self.left)

        if self.bid > 0.0:
            msg = Bid(bid=self.bid, theme=self.theme, user_id=self._current_user_id)
            msgs.append((self.exchange_id, msg))

        return msgs

    def compute_reward(self, _ctx: ph.Context) -> float:
        """@override
        The goal is to maximize the number of clicks so the per-step reward
        is the number of clicks received at the current timestep.
        """
        risk_aversion = 0.0
        return (1 - risk_aversion) * self.step_clicks + (
            risk_aversion * self.left
        ) / self.type.budget

    def is_terminated(self, _ctx: ph.Context) -> bool:
        """@override
        This agent cannot perform any more bids if its budget is 0.
        """
        return self.left <= 0

    def reset(self):
        """@override
        Reset method called before each episode to clear the state of the agent.
        """
        super().reset()

        self.left = self.type.budget

        self.step_clicks = 0
        self.step_wins = 0

        self.total_clicks = defaultdict(int)
        self.total_requests = defaultdict(int)
        self.total_wins = defaultdict(int)

        self.bid = 0.0
        self._current_user_id = 0.0
        self._current_age = 0.0
        self._current_zipcode = 0.0


class AdExchangeAgent(ph.Agent):
    """
    The `AdExchangeAgent` is actually just an actor who reacts to messages reveived.
    It doesn't perform any action on its own.
    """

    @dataclass(frozen=True)
    class AdExchangeView(ph.AgentView):
        """
        The view is used to expose additional information to other actors in the system.
        It is accessible via the `ph.Context` object passed as a parameters
        in the appropriate methods.

        For this use case we want to expose users information to the advertisers to help them
        decide on their bid
        """

        users_info: dict

    def __init__(
        self,
        agent_id: str,
        publisher_id: str,
        advertiser_ids: Iterable = tuple(),
        strategy: str = "first",
    ):
        super().__init__(agent_id)

        self.publisher_id = publisher_id
        self.advertiser_ids = advertiser_ids

        self.strategy = strategy

    def view(self, neighbour_id=None) -> ph.View:
        """@override
        Method to provide extra information about the user. This information
        is made available only for advertisers in a pull fashion, i.e the
        advertiser needs to access the information explicitely via the `ctx`
        object if it wants to use it.
        """
        if neighbour_id and neighbour_id.startswith("ADV"):
            return self.AdExchangeView(
                users_info={
                    1: {"age": 18, "zipcode": 94025},
                    2: {"age": 40, "zipcode": 90250},
                },
            )
        else:
            return super().view(neighbour_id)

    @ph.agents.msg_handler(ImpressionRequest)
    def handle_impression_request(
        self, _ctx: ph.Context, msg: ph.Message[ImpressionRequest]
    ):
        """
        The exchange acts as an intermediary between the publisher and the
        advertisers, upon the reception of an `ImpressionRequest`, the exchange
        simply forward that request to the advertisers
        """
        logger.debug("AdExchange impression request %s", msg)

        return [(adv_id, msg.payload) for adv_id in self.advertiser_ids]

    def handle_batch(
        self,
        ctx: ph.Context,
        batch: Sequence[ph.Message],
    ):
        """@override
        We override the method `handle_batch` to consume all the bids messages
        as one block in order to perform the auction. The batch object contains
        all the messages that were sent to the actor.

        Note:
        -----
        The default logic is to consume each message individually.
        """
        bids = []
        msgs = []
        for message in batch:
            if isinstance(message.payload, Bid):
                bids.append(message)
            else:
                msgs += self.handle_message(ctx, message)

        if len(bids) > 0:
            msgs += self.auction(bids)

        return msgs

    def auction(self, bids: Sequence[ph.Message[Bid]]):
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

        logger.debug("AdExchange auction done winner: %s cost: %s", winner, cost)

        msgs = []

        advertiser_ids = [m.sender_id for m in bids]

        msgs.append(
            (
                self.publisher_id,
                Ads(
                    advertiser_id=winner.sender_id,
                    theme=winner.payload.theme,
                    user_id=winner.payload.user_id,
                ),
            )
        )

        for adv_id in advertiser_ids:
            adv_cost = cost if adv_id == winner.sender_id else 0.0
            msgs.append(
                (
                    adv_id,
                    AuctionResult(cost=adv_cost, winning_bid=winner.payload.bid),
                ),
            )

        return msgs

    def _first_price_auction(self, bids: Sequence[ph.Message[Bid]]):
        sorted_bids = sorted(bids, key=lambda m: m.payload.bid, reverse=True)
        winner = sorted_bids[0]
        cost = sorted_bids[0].payload.bid
        return winner, cost

    def _second_price_auction(self, bids: Sequence[ph.Message[Bid]]):
        sorted_bids = sorted(bids, key=lambda m: m.payload.bid, reverse=True)
        winner = sorted_bids[0]
        cost = (
            sorted_bids[1].payload.bid if len(bids) > 1 else sorted_bids[0].payload.bid
        )
        return winner, cost


#######################################
##  Environment
#######################################


class DigitalAdsEnv(ph.FiniteStateMachineEnv):
    def __init__(self, num_steps=20, num_agents_theme=None, **kwargs):
        # agent ids
        self.exchange_id = "ADX"
        self.publisher_id = "PUB"

        USER_CLICK_PROBABILITIES = {
            1: {"sport": 0.0, "travel": 1.0, "science": 0.2, "tech": 0.5},
            2: {"sport": 1.0, "travel": 0.0, "science": 0.7, "tech": 0.5},
        }

        # The `PublisherAgent`
        publisher_agent = PublisherAgent(
            self.publisher_id,
            exchange_id=self.exchange_id,
            user_click_proba=USER_CLICK_PROBABILITIES,
        )

        # The learning `AdvertiserAgent`s
        advertiser_agents = []
        i = 1
        for theme, num_agents in num_agents_theme.items():
            for _ in range(num_agents):
                advertiser_agents.extend(
                    [
                        AdvertiserAgent(f"ADV_{i}", self.exchange_id, theme=theme),
                    ]
                )
                i += 1
        self.advertiser_ids = [a.id for a in advertiser_agents]

        # The `AdExchangeAgent` that makes the intermediary between publishers and advertisers
        exchange_agent = AdExchangeAgent(
            self.exchange_id,
            publisher_id=self.publisher_id,
            advertiser_ids=self.advertiser_ids,
        )

        # Building the network defining all the actors and connecting them
        actors = [exchange_agent, publisher_agent] + advertiser_agents
        network = ph.StochasticNetwork(
            actors,
            ph.resolvers.BatchResolver(round_limit=5),
            ignore_connection_errors=True,
        )
        network.add_connections_between([self.exchange_id], [self.publisher_id])
        network.add_connections_between([self.exchange_id], self.advertiser_ids)
        network.add_connections_between([self.publisher_id], self.advertiser_ids)

        super().__init__(
            num_steps=num_steps,
            network=network,
            initial_stage="publisher_step",
            stages=[
                ph.FSMStage(
                    stage_id="publisher_step",
                    next_stages=["advertiser_step"],
                    acting_agents=[self.publisher_id],
                    rewarded_agents=[self.publisher_id],
                ),
                ph.FSMStage(
                    stage_id="advertiser_step",
                    next_stages=["publisher_step"],
                    acting_agents=self.advertiser_ids,
                    rewarded_agents=self.advertiser_ids,
                ),
            ],
            **kwargs,
        )


#######################################
##  Metrics
#######################################


class AdvertiserBidUser(ph.metrics.Metric[float]):
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

    def reduce(self, values, mode=None) -> float:
        """@override
        The default logic returns the last step value,
        here we are interested in the average bid value
        """
        return np.nanmean(values)


class AdvertiserAverageHitRatioUser(ph.metrics.Metric[float]):
    def __init__(self, agent_id: str, user_id: int) -> None:
        self.agent_id: str = agent_id
        self.user_id: int = user_id

    def extract(self, env: ph.PhantomEnv) -> float:
        """@override
        Extracts the per-step value to track
        """
        if env[self.agent_id].total_wins[self.user_id] != 0.0:
            return (
                env[self.agent_id].total_clicks[self.user_id]
                / env[self.agent_id].total_wins[self.user_id]
            )
        return np.nan

    def reduce(self, values, mode=None) -> float:
        """@override
        The default logic returns the last step value,
        here we are interested in the average bid value
        """
        return values[-1]


class AdvertiserAverageWinProbaUser(ph.metrics.Metric[float]):
    def __init__(self, agent_id: str, user_id: int) -> None:
        self.agent_id: str = agent_id
        self.user_id: int = user_id

    def extract(self, env: ph.PhantomEnv) -> float:
        """@override
        Extracts the per-step value to track
        """
        if env[self.agent_id].total_requests[self.user_id] != 0.0:
            return (
                env[self.agent_id].total_wins[self.user_id]
                / env[self.agent_id].total_requests[self.user_id]
            )
        return np.nan

    def reduce(self, values, mode=None) -> float:
        """@override
        The default logic returns the last step value,
        here we are interested in the average bid value
        """
        return values[-1]


class AdvertiserTotalRequests(ph.metrics.Metric[float]):
    def __init__(self, agent_id: str, user_id: int) -> None:
        self.agent_id: str = agent_id
        self.user_id: int = user_id

    def extract(self, env: ph.PhantomEnv) -> float:
        return env[self.agent_id].total_requests[self.user_id]

    def reduce(self, values, mode=None) -> float:
        return values[-1]


class AdvertiserTotalWins(ph.metrics.Metric[float]):
    def __init__(self, agent_id: str, user_id: int) -> None:
        self.agent_id: str = agent_id
        self.user_id: int = user_id

    def extract(self, env: ph.PhantomEnv) -> float:
        return env[self.agent_id].total_wins[self.user_id]

    def reduce(self, values, mode=None) -> float:
        return values[-1]


NUM_TRAVEL_ADVERTISERS = 40
NUM_TECH_ADVERTISERS = 40
NUM_SPORT_ADVERTISERS = 40

metrics = {}
for aid in (
    f"ADV_{i}"
    for i in range(
        1,
        NUM_TRAVEL_ADVERTISERS + NUM_TECH_ADVERTISERS + NUM_SPORT_ADVERTISERS + 1,
    )
):
    metrics[f"{aid}/bid_user_1"] = AdvertiserBidUser(aid, 1)
    metrics[f"{aid}/bid_user_2"] = AdvertiserBidUser(aid, 2)
    metrics[f"{aid}/budget_left"] = ph.metrics.SimpleAgentMetric(aid, "left", "mean")
    metrics[f"{aid}/wins"] = ph.metrics.SimpleAgentMetric(aid, "step_wins", "mean")
    metrics[f"{aid}/clicks"] = ph.metrics.SimpleAgentMetric(aid, "step_clicks", "mean")
    metrics[f"{aid}/total_requests_user_1"] = AdvertiserTotalRequests(aid, 1)
    metrics[f"{aid}/total_requests_user_2"] = AdvertiserTotalRequests(aid, 2)
    metrics[f"{aid}/total_wins_user_1"] = AdvertiserTotalWins(aid, 1)
    metrics[f"{aid}/total_wins_user_2"] = AdvertiserTotalWins(aid, 2)
    metrics[f"{aid}/avg_hit_ratio_user_1"] = AdvertiserAverageHitRatioUser(aid, 1)
    metrics[f"{aid}/avg_hit_ratio_user_2"] = AdvertiserAverageHitRatioUser(aid, 2)
    metrics[f"{aid}/avg_win_proba_user_1"] = AdvertiserAverageWinProbaUser(aid, 1)
    metrics[f"{aid}/avg_win_proba_user_2"] = AdvertiserAverageWinProbaUser(aid, 2)

#######################################
##  Params
#######################################

if __name__ == "__main__":
    if sys.argv[1] == "test":
        agent_supertypes = {}
        # travel agency (i.e. agent 1 and 5) have a rather limited budget
        agent_supertypes.update(
            {
                f"ADV_{i}": AdvertiserAgent.Supertype(
                    budget=ph.utils.ranges.LinspaceRange(
                        5.0, 15.0, n=11, name="travel_budget"
                    )
                )
                for i in range(1, NUM_TRAVEL_ADVERTISERS + 1)
            }
        )

        # tech companies have a bigger budget
        agent_supertypes.update(
            {
                f"ADV_{i}": AdvertiserAgent.Supertype(
                    budget=ph.utils.ranges.LinspaceRange(
                        7.0, 17.0, n=11, name="tech_budget"
                    )
                )
                for i in range(
                    NUM_TRAVEL_ADVERTISERS + 1,
                    NUM_TRAVEL_ADVERTISERS + NUM_TECH_ADVERTISERS + 1,
                )
            }
        )

        # sport companies have the bigger budget
        agent_supertypes.update(
            {
                f"ADV_{i}": AdvertiserAgent.Supertype(
                    budget=ph.utils.ranges.LinspaceRange(
                        10.0, 20.0, n=11, name="sport_budget"
                    )
                )
                for i in range(
                    NUM_TRAVEL_ADVERTISERS + NUM_TECH_ADVERTISERS + 1,
                    NUM_TRAVEL_ADVERTISERS
                    + NUM_TECH_ADVERTISERS
                    + NUM_SPORT_ADVERTISERS
                    + 1,
                )
            }
        )

        path = sys.argv[2]
        results = ph.utils.rllib.rollout(
            directory=path,
            num_repeats=50,
            metrics=metrics,
            record_messages=False,
            env_config={
                "agent_supertypes": agent_supertypes,
                "num_agents_theme": {
                    "travel": NUM_TRAVEL_ADVERTISERS,
                    "tech": NUM_TECH_ADVERTISERS,
                    "sport": NUM_SPORT_ADVERTISERS,
                },
            },
        )

        results = list(results)
        cloudpickle.dump(results, open(os.path.join(path, "results.pkl"), "wb"))

        print("Done with evaluation")
    elif sys.argv[1] == "train":
        policies = {
            "adv_policy_travel": [
                f"ADV_{i}" for i in range(1, NUM_TRAVEL_ADVERTISERS + 1)
            ],
            "adv_policy_tech": [
                f"ADV_{i}"
                for i in range(
                    NUM_TRAVEL_ADVERTISERS + 1,
                    NUM_TRAVEL_ADVERTISERS + NUM_TECH_ADVERTISERS + 1,
                )
            ],
            "adv_policy_sport": [
                f"ADV_{i}"
                for i in range(
                    NUM_TRAVEL_ADVERTISERS + NUM_TECH_ADVERTISERS + 1,
                    NUM_TRAVEL_ADVERTISERS
                    + NUM_TECH_ADVERTISERS
                    + NUM_SPORT_ADVERTISERS
                    + 1,
                )
            ],
        }

        agent_supertypes = {}
        # travel agency (i.e. agent 1 and 5) have a rather limited budget
        agent_supertypes.update(
            {
                f"ADV_{i}": AdvertiserAgent.Supertype(
                    budget=ph.utils.samplers.UniformFloatSampler(
                        low=5.0, high=15.0 + 0.001, clip_low=5.0, clip_high=15.0
                    )
                )
                for i in range(1, NUM_TRAVEL_ADVERTISERS + 1)
            }
        )

        # tech companies have a bigger budget
        agent_supertypes.update(
            {
                f"ADV_{i}": AdvertiserAgent.Supertype(
                    budget=ph.utils.samplers.UniformFloatSampler(
                        low=7.0, high=17.0 + 0.001, clip_low=7.0, clip_high=17.0
                    )
                )
                for i in range(
                    NUM_TRAVEL_ADVERTISERS + 1,
                    NUM_TRAVEL_ADVERTISERS + NUM_TECH_ADVERTISERS + 1,
                )
            }
        )

        # sport companies have the bigger budget
        agent_supertypes.update(
            {
                f"ADV_{i}": AdvertiserAgent.Supertype(
                    budget=ph.utils.samplers.UniformFloatSampler(
                        low=10.0,
                        high=20.0 + 0.001,
                        clip_low=10.0,
                        clip_high=20.0,
                    )
                )
                for i in range(
                    NUM_TRAVEL_ADVERTISERS + NUM_TECH_ADVERTISERS + 1,
                    NUM_TRAVEL_ADVERTISERS
                    + NUM_TECH_ADVERTISERS
                    + NUM_SPORT_ADVERTISERS
                    + 1,
                )
            }
        )

        ph.utils.rllib.train(
            algorithm="PPO",
            env_class=DigitalAdsEnv,
            policies=policies,
            metrics=metrics,
            env_config={
                "agent_supertypes": agent_supertypes,
                "num_agents_theme": {
                    "travel": NUM_TRAVEL_ADVERTISERS,
                    "tech": NUM_TECH_ADVERTISERS,
                    "sport": NUM_SPORT_ADVERTISERS,
                },
            },
            iterations=500,
            checkpoint_freq=50,
            rllib_config={
                "seed": 0,
                "batch_mode": "complete_episodes",
                "disable_env_checking": True,
            },
        )
