import mesa
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from generic_model import World

from mesa import Model
from mesa.agent import Agent
from mesa.datacollection import DataCollector


class Agent(mesa.Agent):
    """
    Agent class for the network formation model.
    """

    def __init__(self, model:"World", unique_id):
        super().__init__(unique_id, model)
    
    # ==== Properties ====
    # implement properties of the agent that are useful for the model, especially with respect to its position in the network
    # some are already provided, but you may need to add more
    
    @property
    def uid(self):
        """unique agent identifier, useful for matching agents to network nodes"""
        return self.unique_id

    def subgraph(self, network = None, copy=False):
        """The graph induced by selecting only the component to which the agent belongs."""
        if network is None:
            my_component = self.component_uids()
            network = self.model.net
        else:
           my_component = self.component_uids(network)

        sub_graph = nx.subgraph(network, my_component)
        if copy:
            return sub_graph.copy()
        return sub_graph
    
    def component_uids(self, network = None):
        """returns the uids of the component in which this agent resides"""
        if network is None:
            network = self.model.net
        my_comp = [c for c in nx.connected_components(network) if self.uid in c][0]
        return my_comp
    
    def utility(self):
        pass

    def perform_best_action(self):
        pass

    def step(self):
        max_sample = self.model.max_sample
        mutual_create = self.model.mutual_create
        mutual_delete = self.model.mutual_delete

        # perform the best action available
        self.perform_best_action(mutual_create, mutual_delete, max_sample)