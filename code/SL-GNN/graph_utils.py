import torch
import networkx as nx
import random
import copy
import pandas as pd
from itertools import combinations
from visualization_utils import visualize_graph

from params import (
    INFECTED_OPINION,
    AUTHENTIC_OPINION,
)

def update_graph_environment(Graph):
    # update the graph environment
    # source nodes will infect all its neighbors
    # blocked nodes cannot be infected
    infected_nodes = []
    for node in Graph.nodes:
        if Graph.nodes.data('feature')[node][0] <= INFECTED_OPINION:
            infected_nodes.append(node)

    updated_this_iteration = []              # a node can be updated through only one source node
    for node in infected_nodes:
        for neighbor in Graph.neighbors(node):
            if (neighbor in updated_this_iteration or
                    Graph.nodes.data('feature')[neighbor][0] >= AUTHENTIC_OPINION or
                    Graph.nodes.data('feature')[neighbor][0] <= INFECTED_OPINION):
                continue
            if (Graph.nodes.data('feature')[neighbor][0] > INFECTED_OPINION and
                    Graph.nodes.data('feature')[neighbor][0] < AUTHENTIC_OPINION):
                diff = Graph.nodes.data('feature')[node][0] - Graph.nodes.data('feature')[neighbor][0]
                increment_factor = Graph.edges[node, neighbor]['weight'] * diff
                Graph.nodes[neighbor]['feature'][0] = min(1, max(-1, (Graph.nodes[neighbor]['feature'][0] + increment_factor)))
                updated_this_iteration.append(neighbor)

    # degree of the nodes will be updated
    # if node is connected to a blocked node or source node, its degree will be lower than actual degree
    for node in Graph.nodes:
        Graph.nodes[node]['feature'][1] = Graph.degree[node]

        for neighbor in Graph.neighbors(node):
            if (Graph.nodes.data('feature')[neighbor][0] >= AUTHENTIC_OPINION
                    or Graph.nodes.data('feature')[neighbor][0] <= INFECTED_OPINION):
                Graph.nodes[node]['feature'][1] -= 1

    # calculating the shortest path from each node using a bfs
    def bfs(Graph, current_node):
        # if the path is through a blocked node, we need to check if there is another shortest path
        # if there is no other shortest path, the shortest path length will be updated to 20
        visited_nodes = set()
        visited_nodes.add(current_node)
        queue = [current_node]
        distance = dict()
        distance[current_node] = 0
        shortest_path_length = Graph.number_of_nodes()

        while queue:
            node = queue.pop(0)
            if Graph.nodes.data('feature')[node][0] > INFECTED_OPINION:
                if distance[node] < shortest_path_length:
                    shortest_path_length = distance[node]
                continue
            for neighbor in Graph.neighbors(node):
                if neighbor not in visited_nodes and Graph.nodes.data('feature')[neighbor][0] != AUTHENTIC_OPINION:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
                    distance[neighbor] = distance[node] + 1
        return shortest_path_length

    # calculating the shortest path from each node using a bfs
    for node in Graph.nodes:
        if Graph.nodes[node]['feature'][0] <= INFECTED_OPINION:
            Graph.nodes[node]['feature'][2] = 0
            continue
        elif Graph.nodes[node]['feature'][0] == AUTHENTIC_OPINION:
            Graph.nodes[node]['feature'][2] = Graph.number_of_nodes()
        Graph.nodes[node]['feature'][2] = bfs(copy.deepcopy(Graph), node)

    return Graph

def get_graph_properties(Graph):
    """
    :param Graph: NetworkX Graph
    :return: number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes
    """

    infected_nodes = []
    blocked_nodes = []
    uninfected_nodes = []
    number_of_nodes = Graph.number_of_nodes()

    # -1: infected opinion
    # 0: neutral opinion
    # 1: authentic opinion   ---> blocked nodes
    for node in Graph.nodes:
        if (Graph.nodes.data('feature')[node][0] < AUTHENTIC_OPINION
                and Graph.nodes.data('feature')[node][0] > INFECTED_OPINION):
            uninfected_nodes.append(node)
        if Graph.nodes.data('feature')[node][0] >= AUTHENTIC_OPINION:
            blocked_nodes.append(node)
        if Graph.nodes.data('feature')[node][0] <= INFECTED_OPINION:
            infected_nodes.append(node)

    return number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes


def get_graph_features(Graph):
    node_features = Graph.nodes.data('feature')
    node_features = torch.tensor([node_feature[1] for node_feature in node_features], dtype=torch.float)
    edge_index = torch.tensor(list(Graph.edges), dtype=torch.int64).t().contiguous()
    edge_weights = [Graph[edge[0]][edge[1]]['weight'] for edge in Graph.edges()]
    weights_tensor = torch.tensor(edge_weights, dtype=torch.float)

    return node_features, edge_index, weights_tensor

def generate_graph(num_nodes, case=1):
    """
    :param num_nodes: Number of Nodes
    :param graph_type: Graph Type
    :param case: Three cases:
        1. Node feature[0] is binary, edge weights are 1
        2. Node feature[0] is floating point value, edge weights are 1
        3. Node feature[0] is floating point value, edge weights are between 0 to 1
    :return: Graph, node_features, edge_index, source_node, weights_tensor
    """

    Graph = None
    k = 3
    p = 0.4
    Graph = nx.watts_strogatz_graph(num_nodes, k, p)

    source_node = random.randint(1, 3)
    shortest_paths = nx.shortest_path_length(Graph, source=source_node)
    Graph.graph['num_sources'] = 1

    # Assign random values between 0 and 1 to the edges
    for edge in Graph.edges():
        if case in [1, 2]:
            Graph[edge[0]][edge[1]]['weight'] = 1.0
        else:
            Graph[edge[0]][edge[1]]['weight'] = random.uniform(0, 1)

    if case == 1:
        for node in Graph.nodes:
            Graph.nodes[node]['feature'] = [
                -1 if node == source_node else 0,
                Graph.degree[node],
                shortest_paths.get(node, Graph.number_of_nodes())
            ]
    elif case == 2 or case == 3:
        for node in Graph.nodes:
            Graph.nodes[node]['feature'] = [
                -1 if node == source_node else random.uniform(-0.90, 1),
                Graph.degree[node],
                shortest_paths.get(node, Graph.number_of_nodes())
            ]

    node_features, edge_index, weights_tensor = get_graph_features(Graph)

    return Graph, node_features, edge_index, source_node, weights_tensor


def simulate_propagation(Graph):
    # run dfs from every infected nodes
    # have a list of infected nodes

    number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)
    infected_nodes = set(infected_nodes)
    visited_nodes = set()

    while infected_nodes:
        # visualize_graph(Graph)
        current_node = infected_nodes.pop()
        visited_nodes.add(current_node)
        neighbors = Graph.neighbors(current_node)

        for neighbor in neighbors:
            if neighbor not in visited_nodes and neighbor not in blocked_nodes:
                edge_weight = Graph[current_node][neighbor]['weight']
                neighbor_opinion = Graph.nodes.data('feature')[neighbor][0]
                current_node_opinion = -1 # infectious opinion
                new_value = neighbor_opinion + edge_weight * (current_node_opinion - neighbor_opinion)

                if new_value <= INFECTED_OPINION:
                    infected_nodes.add(neighbor)

                Graph.nodes[neighbor]['feature'][0] = new_value

    number_of_nodes, infected_nodes, _, _ = get_graph_properties(Graph)
    num_infected_nodes = len(set(infected_nodes))
    proportion_infected_nodes = num_infected_nodes / number_of_nodes

    return proportion_infected_nodes


def output_with_minimal_infection_rate(Graph, budget=1):
    _, _, _, uninfected_nodes = get_graph_properties(copy.deepcopy(Graph))
    minimal_infection_rate = 1
    minimal_infection_rate_output = None
    list_of_nodes_to_block = None

    if budget >= len(uninfected_nodes):
        minimal_infection_rate_output = torch.zeros((Graph.number_of_nodes(), 1), requires_grad=False)

        for node in uninfected_nodes:
            minimal_infection_rate_output[node] = 1

        return minimal_infection_rate_output, uninfected_nodes

    else:
        all_combinations = list(combinations(uninfected_nodes, budget))
        all_combinations_as_lists = [list(combination) for combination in all_combinations]

        for combination in all_combinations_as_lists:
            simulation_graph = copy.deepcopy(Graph)
            for node in combination:
                simulation_graph.nodes[node]['feature'][0] = AUTHENTIC_OPINION

            infection_rate = simulate_propagation(copy.deepcopy(simulation_graph))

            if infection_rate <= minimal_infection_rate:
                minimal_infection_rate = infection_rate
                minimal_infection_rate_output = torch.zeros((Graph.number_of_nodes(), 1), requires_grad=False)
                for node in combination:
                    minimal_infection_rate_output[node] = 1
                list_of_nodes_to_block = combination

    return minimal_infection_rate_output, list_of_nodes_to_block