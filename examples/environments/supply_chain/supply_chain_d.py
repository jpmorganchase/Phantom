import csv
import cloudpickle
#import pickle
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gym
import numpy as np
import phantom as ph
import random
random.seed(100)
#torch.manual_seed(200)
np.random.seed(300)

# Define fixed parameters:
NUM_EPISODE_STEPS = 60 #100 
NUM_SHOPS = 1
NUM_CUSTOMERS = 1#5 

CUSTOMER_MAX_ORDER_SIZE = 26
SHOP_MIN_PRICE = 0.0
SHOP_MAX_PRICE = 2.0
SHOP_MAX_STOCK = 100
SHOP_MAX_STOCK_REQUEST = int(CUSTOMER_MAX_ORDER_SIZE * NUM_CUSTOMERS * 1.5)
max_initial_inv = 26
ALLOW_PRICE_ACTION = False


@dataclass(frozen=True)
class OrderRequest(ph.MsgPayload):
    """Customer --> Shop"""

    size: int


@dataclass(frozen=True)
class OrderResponse(ph.MsgPayload):
    """Shop --> Customer"""

    size: int


@dataclass(frozen=True)
class StockRequest(ph.MsgPayload):
    """Shop --> Factory"""

    size: int


@dataclass(frozen=True)
class StockResponse(ph.MsgPayload):
    """Factory --> Shop"""

    size: int


class CustomerPolicy(ph.Policy):
    def compute_action(self, obs: np.ndarray) -> Tuple[int, int]:
        return (np.random.randint(CUSTOMER_MAX_ORDER_SIZE), np.argmin(obs))

class CustomerAgent(ph.MessageHandlerAgent):
    def __init__(self, agent_id: ph.AgentID, shop_ids: List[ph.AgentID]):
        super().__init__(agent_id)

        # We need to store the shop IDs so we can send order requests to them.
        self.shop_ids: List[str] = shop_ids

        self.action_space = gym.spaces.Tuple(
            (
                # The number of items to order
                gym.spaces.Discrete(CUSTOMER_MAX_ORDER_SIZE),
                # The shop to order from
                gym.spaces.Discrete(len(self.shop_ids)),
            )
        )

        # The price set by each shop
        self.observation_space = gym.spaces.Box(
            low=SHOP_MIN_PRICE, high=SHOP_MAX_PRICE, shape=(len(self.shop_ids),)
        )

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(self, ctx: ph.Context, message: ph.Message):
        return

    def decode_action(self, ctx: ph.Context, action: Tuple[int, int]):
        # At the start of each step we generate an order with a random size to send to a
        # random shop.
        order_size, selected_shop = action
        shop_id = self.shop_ids[selected_shop]
        return [(shop_id, OrderRequest(order_size))]

    def encode_observation(self, ctx: ph.Context):
        return np.array(
            [ctx[shop_id].price for shop_id in self.shop_ids], dtype=np.float32
        )

    def compute_reward(self, ctx: ph.Context) -> float:
        return 0.0


class FactoryAgent(ph.MessageHandlerAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    @ph.agents.msg_handler(StockRequest)
    def handle_stock_request(self, ctx: ph.Context, message: ph.Message):
        return [(message.sender_id, StockResponse(message.payload.size))]


class ShopAgent(ph.MessageHandlerAgent):
    @dataclass
    class Supertype(ph.Supertype):
        #sale_price: float
        cost_of_carry: float
        #cost_per_unit: float
    
    @dataclass(frozen=True)
    class View(ph.AgentView):
        price: float
    

    def __init__(self, agent_id: str, factory_id: str):
        super().__init__(agent_id)
        self.factory_id: str = factory_id
        self.price: float = SHOP_MAX_PRICE

        
        self.sales: int = 0   #have to be 0 at every time-step
        self.missed_sales: int = 0 #have to be 0 at every time-step
        self.delivered_stock: int = 0 #have to be 0 at every time-step
        self.orders_received: int = 0 #have to be 0 at every time-step
        #self.total_stock: int = 0 #have to be 0 at every time-step
        self.total_stock: int = np.random.randint(max_initial_inv)
        self.leftover: int = 0 #does not have to be 0 at every time-step
        self.stock: int = 0   #does not have to be 0 at every time-step
        
        
        
        
    
        # We initialise the price variable here, it's value will be set when the shop
        # agent takes it's first action.
        
      


    @property
    def action_space(self):
        if ALLOW_PRICE_ACTION:
            return gym.spaces.Dict(
                {
                    # The price to set for the current step:
                    "price": gym.spaces.Box(
                        low=SHOP_MIN_PRICE, high=SHOP_MAX_PRICE, shape=(1,)
                    ),
                    # The number of additional units to order from the factory:
                    "restock": gym.spaces.Discrete(SHOP_MAX_STOCK_REQUEST),
                }
            )
        else:
            return gym.spaces.Dict(
                {
                    {
                    # "restock_qty": gym.spaces.Discrete(SHOP_MAX_STOCK_REQUEST),
                    "restock_qty": gym.spaces.Box(
                        low=0, high=SHOP_MAX_STOCK_REQUEST, shape=(1,)
                    ),
                }
                }
            )

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                # The agent's type:
                "type": self.type.to_obs_space(),
                # The agent's current stock:
                "stock": gym.spaces.Box(low=0, high=1, shape=(1,)),

                #"stock": gym.spaces.Discrete(SHOP_MAX_STOCK + 1),
                # The number of sales made by the agent in the previous step:
                 "previous_sales": gym.spaces.Box(low=0, high=1, shape=(1,)),

                "previous_cusorders": gym.spaces.Box(low=0, high=1, shape=(1,)),
            }
        )
    
    def view(self, neighbour_id: Optional[ph.AgentID] = None) -> "View":
        """Return an immutable view to the agent's public state."""
        return self.View(self.id, self.price)
    

    def pre_message_resolution(self, ctx: ph.Context):
        
        if ctx["ENV"].stage == "sales_step":
            self.sales = 0   #have to be 0 at every time-step
            self.missed_sales = 0 #have to be 0 at every time-step
            #self.delivered_stock = 0 #have to be 0 at every time-step new edit
            self.orders_received = 0 #have to be 0 at every time-step
            #self.total_stock = 0 #have to be 0 at every time-step new edit
            #self.stock = 0 new edit
            
            # At the start of each step we reset the number of missed orders to 0.  #is this each time-step or reset after one episode
        
            

    @ph.agents.msg_handler(StockResponse)
    def handle_stock_response(self, ctx: ph.Context, message: ph.Message):
        #print("22 leftover stock from prev step:", self.total_stock)#self.leftover_stock)
        self.delivered_stock = message.payload.size

        with open('restock.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.delivered_stock])


        self.total_stock = min((self.total_stock + message.payload.size), SHOP_MAX_STOCK) 
        
       

    @ph.agents.msg_handler(OrderRequest)
    def handle_order_request(self, ctx: ph.Context, message: ph.Message):
        #self.orders_received += 1

        amount_requested = message.payload.size
        amount_requested = self.orders_received


        with open('customerorders.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([amount_requested])

        with open('totalstock.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.total_stock])
     

 
        # If the order size is more than the amount of stock, partially fill the order.
        if amount_requested > self.total_stock:
            self.missed_sales = amount_requested - self.total_stock#+= amount_requested - self.stock
            stock_to_sell = self.total_stock   #this is the total stock in sale step
            self.total_stock = 0
        # ... Otherwise completely fill the order.
        else:
            stock_to_sell = amount_requested
            self.total_stock =  self.total_stock - amount_requested #-= amount_requested
            self.missed_sales = 0

        self.sales = stock_to_sell #+= stock_to_sell
        #print("sales:", self.sales)
   
        with open('sales.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.sales])


        with open('missedsales.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.missed_sales])


        self.leftover = self.total_stock 
        with open('leftover.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.leftover])

        
        # Send the customer their order.
        return [(message.sender_id, OrderResponse(stock_to_sell))]

    def encode_observation(self, ctx: ph.Context):
      
        return {
            # The shop's type is included in its observation space to allow it to learn
            # a generalised policy.
            "previous_sales": np.array([self.sales])/25, #min max normalization

            "stock": np.array([self.total_stock])/SHOP_MAX_STOCK,

            "previous_cusorders": np.array([self.metcusorders])/25, 

            "type": self.type.to_obs_space_compatible_type(),
            
        }

    def decode_action(self, ctx: ph.Context, action: Tuple[np.ndarray, int]):
       
        with open('Actions.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([StockRequest(int(action["restock_qty"][0]))])
        return [(self.factory_id, StockRequest(int(action["restock_qty"][0])))]
        

    def compute_reward(self, ctx: ph.Context) -> float:
        #print("Rew - sales, restock, leftover:", self.sales, self.delivered_stock, self.total_stock) #self.leftover_stock)
        #print("cc:", self.type.cost_of_carry)
        rr = (self.sales * 1.0 - self.delivered_stock * 0.5 - self.total_stock * self.type.cost_of_carry)
        with open('rewards.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rr])
        return rr
            # The shop makes profit from selling items at the set price:
             #self.price
            # It incurs a cost for ordering new stock:
             #self.type.cost_per_unit
            # And for holding onto leftover stock overnight:
            
        
        
        

    def reset(self):
        super().reset()  # sampled supertype is set as self.type here
        self.total_stock = np.random.randint(max_initial_inv)

    
        
 


# Define agent IDs:
FACTORY_ID = "FACTORY"
SHOP_IDS = [f"SHOP{i+1}" for i in range(NUM_SHOPS)]
CUSTOMER_IDS = [f"CUST{i+1}" for i in range(NUM_CUSTOMERS)]


class SupplyChainEnv(ph.FiniteStateMachineEnv):
    def __init__(self, **kwargs):
        shop_agents = [ShopAgent(id, FACTORY_ID) for id in SHOP_IDS]

        factory_agent = FactoryAgent(FACTORY_ID)

        customer_agents = [CustomerAgent(id, shop_ids=SHOP_IDS) for id in CUSTOMER_IDS]

        agents = [factory_agent] + shop_agents + customer_agents

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the shops to the factory
        network.add_connections_between(SHOP_IDS, [FACTORY_ID])

        # Connect the shop to the customers
        network.add_connections_between(SHOP_IDS, CUSTOMER_IDS)

        super().__init__(
            num_steps=NUM_EPISODE_STEPS,
            network=network,
            initial_stage="sales_step",
            stages=[
                ph.FSMStage(
                    stage_id="sales_step",
                    next_stages=["restock_step"],
                    acting_agents=CUSTOMER_IDS,
                    rewarded_agents=SHOP_IDS,   # [],
                ),
                ph.FSMStage(
                    stage_id="restock_step",
                    next_stages=["sales_step"],
                    acting_agents=SHOP_IDS,
                    rewarded_agents=[],  #SHOP_IDS,
                ),
            ],
            **kwargs,
        )


metrics = {}

for id in SHOP_IDS:
    metrics[f"leftover/{id}"] = ph.logging.SimpleAgentMetric(id, "leftover", "mean") #LEFTOVER
    metrics[f"delivered_stock/{id}"] = ph.logging.SimpleAgentMetric(id, "delivered_stock", "mean") #RESTOCK
    metrics[f"total_stock/{id}"] = ph.logging.SimpleAgentMetric(id, "total_stock", "mean") #TOTAL STOCK
    metrics[f"orders_received/{id}"] = ph.logging.SimpleAgentMetric(id, "orders_received", "mean") #CUSTOMER ORDERS
    metrics[f"sales/{id}"] = ph.logging.SimpleAgentMetric(id, "sales", "mean")  #SALES
    metrics[f"missed_sales/{id}"] = ph.logging.SimpleAgentMetric(id, "missed_sales", "mean") #MISSED SALES
   




if sys.argv[1] == "train":
    ph.utils.rllib.train(
        algorithm="PPO",
        num_workers=1,
        env_class=SupplyChainEnv,
        env_config={
            "agent_supertypes": {
                shop_id: ShopAgent.Supertype(
                   cost_of_carry=ph.utils.samplers.UniformFloatSampler(low=0.0, high=1.0),
                )
                for shop_id in SHOP_IDS
            }
        },
        policies={
            "shop_policy": ShopAgent, 
            "customer_policy": (CustomerPolicy, CustomerAgent),
        },
        policies_to_train=["shop_policy"],
        metrics=metrics,
        rllib_config={
            "seed": 123,
            # "model": {
            #     "fcnet_hiddens": [32, 32],
            # },
            "disable_env_checking": True,
        },
        tune_config={
            # "name": "fcnet_32_32",
            "checkpoint_freq": 100,  #100,
            "stop": {
                "training_iteration": 5000,#10000,
            },
        },
    )



elif sys.argv[1] == "test":
    results = ph.utils.rllib.rollout(
        directory="PPO/LATEST",
        algorithm="PPO",
        env_class=SupplyChainEnv,
        env_config={
            "agent_supertypes": {
                shop_id: ShopAgent.Supertype(
                    # cost_of_carry=ph.utils.ranges.UniformRange(
                    #     start=0.0,
                    #     end=0.1 + 0.001,
                    #     step=0.01,
                    # ),
                    # cost_per_unit=ph.utils.ranges.UniformRange(
                    #     start=0.2,
                    #     end=0.8 + 0.001,
                    #     step=0.05,
                    # ),
                    #sale_price=1.0,
                    cost_of_carry=0.2,
                    #cost_per_unit=0.5,
                )
                for shop_id in SHOP_IDS
            }
        },
        num_repeats=1,
        metrics=metrics,
        record_messages=False,
        num_workers=0,
    )

    # pickle.dump(results, open("results.pkl", "wb"))

    print([x for x in results[0].actions_for_agent("SHOP1") if x])
    print("len1:", len([x for x in results[0].actions_for_agent("SHOP1") if x]))
    #print(results[0].metrics["sales/SHOP1"])
    print(results[0].metrics["metmissedsales/SHOP1"])
    print(results[0].metrics["metsales/SHOP1"])
    print(results[0].metrics["metrestock/SHOP1"])
    print(results[0].metrics["leftover/SHOP1"])
    print(results[0].metrics["mettotalstock/SHOP1"])
    print("len2:", len(results[0].metrics["metmissedsales/SHOP1"]))

'''
elif sys.argv[1] == "test":
    def fn(rollout):
        return {
           "cost_of_carry": rollout.env_config["agent_supertypes"][
                "SHOP1"
        ].cost_of_carry,
        #"mean_restock":np.mean(rollout.metrics["metrestock/SHOP1"]),
        "mean_restock":rollout.metrics["metrestock/SHOP1"],

        #"mean_restock":np.mean(rollout.metrics["metrestock/SHOP1"]),
        #"mean_restock":rollout.metrics["metrestock/SHOP1"],

        #"mean_leftover":np.mean(rollout.metrics["leftover/SHOP1"]),
        "mean_leftover":rollout.metrics["leftover/SHOP1"],

        #"mean_sales":np.mean(rollout.metrics["metsales/SHOP1"]),
        "mean_sales":rollout.metrics["metsales/SHOP1"],

        #"mean_customer_orders":np.mean(rollout.metrics["metcusorders/SHOP1"]),
        "mean_customer_orders":rollout.metrics["metcusorders/SHOP1"],

        #"mean_missed_sales":np.mean(rollout.metrics["metmissedsales/SHOP1"]),
        "mean_missed_sales":rollout.metrics["metmissedsales/SHOP1"],

        #"mean_total_stock":np.mean(rollout.metrics["mettotalstock/SHOP1"]),
        "mean_total_stock":rollout.metrics["mettotalstock/SHOP1"],

        }

    
    results = ph.utils.rllib.rollout(
        #directory="PPO/LATEST",
        directory="PPO/PPO_SupplyChainEnv_51cc0_00000_0_2022-09-21_22-47-07",
        algorithm="PPO",
        env_class=SupplyChainEnv,
        env_config={
            "agent_supertypes": {
                shop_id: ShopAgent.Supertype(
                    cost_of_carry = 0.0,
                    # cost_of_carry=ph.utils.ranges.UniformRange(
                    #     start=0.0,
                    #     end=0.1 + 0.001,
                    #     step=0.01,
                    # ),
                    # cost_per_unit=ph.utils.ranges.UniformRange(
                    #     start=0.2,
                    #     end=0.8 + 0.001,
                    #     step=0.05,
                    # ),
                   
                    #cost_of_carry=ph.utils.ranges.UniformRange(
                    #start=0.0,
                    #end=0.2 + 0.005,   #0.9999
                    #step=0.001,
                #),
                
                )
                for shop_id in SHOP_IDS
            }
        },
        num_repeats=1,  #100, #500
        #num_workers=0,
        num_workers=0, #1
        metrics=metrics,
        record_messages=False,
        result_mapping_fn=fn,
        
    )

    #cloudpickle.dump(list(results), open("resultsnew27perepisode1.pkl", "wb"))
    print("evaluation done")
'''

