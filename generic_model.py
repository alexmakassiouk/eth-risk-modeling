import mesa
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random

from mesa import Model
from mesa.agent import Agent
from mesa.datacollection import DataCollector


class World(mesa.Model):
    """The World class represents the environment in which the Agents interact.

    The class keeps track of the Network on which the agents interact, the alterations to it
    over time as well as the model time
    """

    def __init__(self, num_agents, cost,
                 mutual_create, 
                 mutual_delete,
                 poling_interval=1, 
                 check_stability=None, 
                 max_steps=None,
                 max_sample=None,
                 plot_network=False,
                 schedule=mesa.time.BaseScheduler,
                 alpha=1.0
                 ):
        super().__init__()
        self.schedule = schedule(self)
        self.net = nx.Graph()
        self.max_sample = max_sample
        self.cost = cost
        self._id_counter = 0
        self.poling_interval = poling_interval
        self.check_stability = check_stability
        if not self.check_stability and max_steps:
            self.check_stability = max_steps + 1
        else:
            self.check_stability = 1
        self.mutual_create = mutual_create
        self.mutual_delete = mutual_delete
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.plot = plot_network
        self.alpha = alpha

        self.add_agents(num_agents)

        self.datacollector = DataCollector(
            {
                "welfare": lambda m: m.aggregate_welfare(),
                "density": lambda m: m.density(),
                "clustering": lambda m: m.clustering(),
                "num_components": lambda m: len(list(m.connected_components())),
            }
        )

    # === Basic Actions ===

    def _generate_uid(self):
        """generate a unique uid"""
        valid_id_found = False
        uid = None
        while not valid_id_found:
            uid = self._id_counter
            valid_id_found = uid not in self.net
            self._id_counter += 1
        return uid

    def add_agents(self, num_agents):
        """Add the specified number of agents or ids in agent_list to the World.
        """
        for _ in range(num_agents):
            self.add_agent()

    def add_agent(self):
        """Add a single Agent to the simulation
        """
        uid = self._generate_uid() # generate a unique uid for the agent in the network
        if uid in self.net:
            raise KeyError(f"`uid={uid}` already taken")

        agent = Agent(self, uid)
        self.schedule.add(agent)
        self.net.add_node(uid)

    # === Network properties (for reporting) ===
    def aggregate_welfare(self):
        """Compute aggregated welfare (i.e. the sum of all utilities) of the agents in the network
        """
        welfare = sum(a.utility(a.subgraph()) for a in self.schedule.agents)
        return welfare

    def density(self):
        """Returns the proportion of possible edges which are actually present"""
        return nx.density(self.net)

    def clustering(self):
        """See networkx.average_clustering"""
        return nx.average_clustering(self.net)

    def connected_components(self):
        """Returns a generator containing the uids of the current network

        Yields:
            sets of uids in the same component
        """
        return nx.connected_components(self.net)

    # === Stability Checks (to stop simulation) ===
    def is_pairwise_stable(self, agents=None, mutual_create=True, mutual_delete=True):
        """Return true if no agent would be willing to make a change (either create or delete)

        Args:
            agents (list of Agents): if the stability should only be checked for this subset pass a
                list of agents otherwise the stability is computed considering all agents
            mutual_create (bool): require that neither agent is worse off when creating the link
            mutual_delete (bool): require that neither agent is worse off when deleting the link

        Returns:
            Returns True if no agent wants to make a move, False otherwise
        """
        # short circuit logic to reduce the number of comparisons necessary to find that the
        # system is not stable (i.e. at least one agent wants to make a move).
        # If it is in fact stable all agents need to be tested
        deletion_stable = self.is_deletion_stable(agents, mutual=mutual_delete)

        if deletion_stable is False:  # the first test is False -> all is False
            return False

        # The first test has yielded True if the next test is False -> all is False
        creation_stable = self.is_creation_stable(agents, mutual=mutual_create)
        return creation_stable

    def is_deletion_stable(self, agents=None, mutual=True):
        """Returns True if no Agent would be willing to delete a link"""
        return self._is_simple_action_stable(agents, create=False, mutual=mutual)

    def is_creation_stable(self, agents=None, mutual=True):
        """Returns True if no Agent would be willing to create one more edge"""
        return self._is_simple_action_stable(agents, create=True, mutual=mutual)

    def _is_simple_action_stable(self, agents=None, create=True, mutual=True):
        """Given either create=True or create=False (i.e. delete) the function returns True if no
        Agent would be willing to perform an action of the given type."""
        if agents is None:
            agents = self.schedule.agents

        for agent in agents:
            _, other_agent = agent.best_simple_action(create, mutual)
            if other_agent is not None:
                return False
        return True
    
    def step(self):
        # at given interval report on the state of the system
        if self.schedule.time % self.poling_interval == 0:
            self.datacollector.collect(self)
            
        self.schedule.step()
        check_stability = self.check_stability
        mutual_create = self.mutual_create
        mutual_delete = self.mutual_delete
    

        if self.schedule.time>0 and check_stability and (self.schedule.time % check_stability == 0):
            print(check_stability)
            if self.is_pairwise_stable(
                mutual_create=mutual_create, mutual_delete=mutual_delete
            ):
                self.running = False
        
        if self.max_steps and self.schedule.time >= self.max_steps:
            self.running = False