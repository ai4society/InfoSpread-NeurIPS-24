import numpy as np
import random
import networkx as nx
import torch.nn as nn
# from model import ResNet

from typing import List, Tuple

class MisInfoSpreadState:
    def __init__(self, node_states: np.ndarray, adjacency_matrix: np.ndarray, time_step: int):
        self.node_states = node_states
        self.adjacency_matrix = adjacency_matrix
        self.time_step = time_step

    def __hash__(self):
        return hash(str(self.node_states))

    def __eq__(self, other):
        return np.array_equal(self.node_states, other.node_states)


class MisInfoSpread:
    INFECTED_VALUE = -1

    def __init__(self, num_nodes, max_time_steps, trust_on_source=1, 
                 positive_info_threshold=0.95, count_infected_nodes = 1, count_actions = 1):
        self.num_nodes = num_nodes
        self.max_time_steps = max_time_steps

        self.trust_on_source = trust_on_source
        self.positive_info_threshold = positive_info_threshold
        self.count_infected_nodes = count_infected_nodes
        self.count_actions = count_actions

    def find_neighbor(self, state):
        neighbors = set()
        for i, node_value in enumerate(state.node_states):
            if node_value == self.INFECTED_VALUE:
                for j, connection in enumerate(state.adjacency_matrix[i]):
                    if connection != 0 and state.node_states[j] != self.INFECTED_VALUE and state.node_states[j] != 1:
                        neighbors.add(j)
        return list(neighbors)

    def find_neighbor_batch(self, states):
        return [self.find_neighbor(state) for state in states]

    def step(self, state, action_list):
        next_state, _ = self.next_state([state], action_list)
        done = not self.find_neighbor(next_state[0]) or next_state[0].time_step >= self.max_time_steps
        return next_state[0], self.reward([state], [next_state[0]])[0], done

    def step_batch(self, states: List[MisInfoSpreadState], actions) -> Tuple[List[MisInfoSpreadState], List[float], List[bool]]:
        results = [self.step(state, action_list) for state, action_list in zip(states, actions)]
        next_states, rewards, dones = zip(*results)
        return list(next_states), list(rewards), list(dones)

    def find_connections(self, nodeID, state):
        connections = []
        for i, connection in enumerate(state.adjacency_matrix[nodeID]):
            if connection != 0:
                connections.append(i)
        return connections

    def next_state(self, states: List['MisInfoSpreadState'], action_list: List[int]) -> Tuple[List['MisInfoSpreadState'], List[float]]:
        next_states, costs = [], []
        for state in states:

            current_node_states = state.node_states.copy()
            adjacency_matrix = state.adjacency_matrix
            current_time_step = state.time_step

            if len(action_list) > 0:

                for action in action_list:
                    if current_node_states[action] != self.INFECTED_VALUE:
                        current_node_states[action] = 1

                neighbors = self.find_neighbor(state)

                for candidate_node in neighbors:
                    if candidate_node not in action_list:
                        current_node_states[candidate_node] = self.INFECTED_VALUE

            new_state = MisInfoSpreadState(current_node_states, adjacency_matrix, current_time_step + 1)
            next_states.append(new_state)
            costs.append(1)  # Add relevant cost calculation here

        return next_states, costs

    # def reward(self, states: List['MisInfoSpreadState']) -> List[float]:
    #     rewards = []
    #     for state in states:
    #         inf_rate = -(state.node_states.count(self.INFECTED_VALUE)/self.num_nodes)
    #         rewards.append(inf_rate)
        
    #     return rewards

    # def reward(self, states: List['MisInfoSpreadState'], next_states: List['MisInfoSpreadState']) -> List[float]:
    #     rewards = []
    #     for state, next_states in zip(states, next_states):
    #         inf_rate = -((next_states.node_states.count(self.INFECTED_VALUE)/self.num_nodes) - (state.node_states.count(self.INFECTED_VALUE)/self.num_nodes))*10
    #         # rewards.append(inf_rate)

    #         done = not self.find_neighbor(next_states)
    #         if done:
    #             rewards.append( inf_rate + 1 )
    #         else:
    #             rewards.append( inf_rate )

    #     return rewards

    def reward(self, states: List['MisInfoSpreadState'], next_states: List['MisInfoSpreadState']) -> List[float]:
        rewards = []
        for state, next_state in zip(states, next_states):
            inf_nodes = -(next_state.node_states.count(self.INFECTED_VALUE) - state.node_states.count(self.INFECTED_VALUE))
            rewards.append(inf_nodes)

        return rewards

    def get_nnet_model(self) -> nn.Module:
        return ResNet(num_nodes=self.num_nodes, num_blocks=[2, 2, 2, 2])

    def generate_states(self, num_states):

        num_nodes = self.num_nodes
        if num_states <= 0 or num_nodes <= 0:
            raise ValueError("Number of states and nodes must be positive integers.")

        states = []
        for _ in range(num_states):
            state = [0 for _ in range(num_nodes)]

            unique_nodes_to_infect = random.sample(range(num_nodes), self.count_infected_nodes)
            for node_index in unique_nodes_to_infect:
                state[node_index] = self.INFECTED_VALUE

            k = 3
            p = 0.4
            G = nx.watts_strogatz_graph(self.num_nodes, k, p)
            adjacency_matrix = np.array( nx.to_numpy_array(G).tolist() )

            states.append(MisInfoSpreadState(state, adjacency_matrix, 0))

        return states