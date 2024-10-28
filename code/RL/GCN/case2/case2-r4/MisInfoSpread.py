import numpy as np
import random
import networkx as nx
import torch.nn as nn
from model import ResNet
from GNN import GCN

from typing import List, Tuple

class MisInfoSpreadState:
    def __init__(self, node_states: np.ndarray, adjacency_matrix: np.ndarray, 
                 node_features: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray, time_step: int):
        self.node_states = node_states
        self.adjacency_matrix = adjacency_matrix
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_weight = edge_weight
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
                    if connection != 0 and state.node_states[j] != self.INFECTED_VALUE and state.node_states[j] <= self.positive_info_threshold:
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
            current_node_features = state.node_features

            if len(action_list) > 0:

                for action in action_list:
                    current_node_states[action] = round(current_node_states[action] + self.trust_on_source * (1 - current_node_states[action]), 2)
                    current_node_states[action] = min(current_node_states[action], 1)

                state.node_states = current_node_states
                neighbors = self.find_neighbor(state)
                # print("Neighbors: ", neighbors)
                infected_nodes = [i for i, x in enumerate(current_node_states) if x == self.INFECTED_VALUE]

                # Dictionary to store the most influential infected node for each neighbor
                max_connection = {neighbor: (None, -float('inf')) for neighbor in neighbors}
                for i in infected_nodes:
                    for j in neighbors:
                        if adjacency_matrix[i][j] > max_connection[j][1]:
                            max_connection[j] = (i, adjacency_matrix[i][j])

                # Apply infection update based on the most influential infected node
                for neighbor, (infected_node, connection) in max_connection.items():
                    if infected_node is not None:
                        current_node_states[neighbor] = round(current_node_states[neighbor] + connection * (current_node_states[infected_node] - current_node_states[neighbor]), 2)
                        current_node_states[neighbor] = max(min(current_node_states[neighbor], 1), self.INFECTED_VALUE)


            temp_G = nx.from_numpy_array(adjacency_matrix).copy()
            blocked_nodes = []
            for i, node_value in enumerate(current_node_states):
                if node_value >= self.positive_info_threshold:
                    blocked_nodes.append(i)
            temp_G.remove_nodes_from(blocked_nodes)

            infected_nodes = [i for i, node_value in enumerate(current_node_states) if node_value == self.INFECTED_VALUE]
            for idx in range(len(current_node_features)):
                current_node_features[idx][0] = current_node_states[idx]
                if idx in blocked_nodes:
                    current_node_features[idx][1] = 0
                    current_node_features[idx][2] = 9999
                    continue
                
                unq_neighbours = [item for item in temp_G.neighbors(idx) if item not in infected_nodes]
                current_node_features[idx][1] = len(unq_neighbours)

                min_val = 9999
                for node_index in infected_nodes:
                    try:
                        min_val = min(min_val, nx.shortest_path_length(temp_G, idx, node_index))
                    except nx.NetworkXNoPath:
                        pass

                current_node_features[idx][2] = min_val


            # print("Node Features: ", current_node_features)

            # print("State: ", current_node_states)
            # print("Adjacency Matrix: \n", adjacency_matrix)

            new_state = MisInfoSpreadState(current_node_states, adjacency_matrix, current_node_features, state.edge_index, state.edge_weight, current_time_step + 1)
            next_states.append(new_state)
            costs.append(1)  # Add relevant cost calculation here

        return next_states, costs


    def reward(self, states: List['MisInfoSpreadState'], next_states: List['MisInfoSpreadState']) -> List[float]:
        rewards = []
        for state in next_states:
            inf_nodes = -(state.node_states.count(self.INFECTED_VALUE)/self.num_nodes)
            rewards.append(inf_nodes)

        return rewards

    def get_nnet_model(self) -> nn.Module:
        return GCN(input_size=3, hidden_size=64, num_classes=1)

    def generate_states(self, num_states):

        num_nodes = self.num_nodes
        if num_states <= 0 or num_nodes <= 0:
            raise ValueError("Number of states and nodes must be positive integers.")

        states = []
        for _ in range(num_states):
            state = [round(random.uniform(-0.5, 0.6), 2) for _ in range(num_nodes)]

            unique_nodes_to_infect = random.sample(range(num_nodes), self.count_infected_nodes)
            for node_index in unique_nodes_to_infect:
                state[node_index] = self.INFECTED_VALUE

            k = 3
            p = 0.4
            G = nx.watts_strogatz_graph(self.num_nodes, k, p)
            adjacency_matrix = np.array( nx.to_numpy_array(G).tolist() )

            node_features = [[] for _ in range(num_nodes)]
            for i in range(num_nodes):
                node_features[i].append( state[i] )
                neighbours = [item for item in G.neighbors(i) if item not in unique_nodes_to_infect]
                node_features[i].append( len(neighbours) )
                min_val = 9999
                for node_index in unique_nodes_to_infect:
                    try:
                        min_val = min(min_val, nx.shortest_path_length(G, i, node_index))
                    except nx.NetworkXNoPath:
                        pass

                node_features[i].append( min_val )

            edge_index = list(G.edges())
            edge_weight = [adjacency_matrix[i][j] for i, j in edge_index]

            states.append(MisInfoSpreadState(state, adjacency_matrix, node_features, edge_index, edge_weight, 0))

        return states

