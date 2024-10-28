import random

import pandas as pd
import os
from MisInfoSpread import MisInfoSpreadState
import networkx as nx
import torch
from graph_utils import (
    get_graph_properties,
    update_graph_environment,
    get_graph_features,
    generate_graph
)
from GNN import GCN
from params import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
import copy
from tqdm import tqdm
import numpy as np
from params import INFECTED_OPINION, AUTHENTIC_OPINION
from visualization_utils import visualize_graph
from figures import generate_comparison_figures


def create_graph_from_df(df, num_nodes, case=1):
    Graph = nx.Graph()
    Graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if df.adjacency_matrix[i][j] > 0:
                Graph.add_edge(i, j)

    for edge in Graph.edges():
        if case == 1 or case == 2:
            Graph[edge[0]][edge[1]]['weight'] = 1.0
        else:
            Graph[edge[0]][edge[1]]['weight'] = df.adjacency_matrix[edge[0]][edge[1]]

    source_nodes = []
    for node, value in enumerate(df.node_states):
        if value <= INFECTED_OPINION:
            source_nodes.append(node)

    if case == 1:
        for node in Graph.nodes:
            Graph.nodes[node]['feature'] = [
                -1 if node in source_nodes else 0,
                Graph.degree[node],
                min(nx.shortest_path_length(Graph, source=source_node).get(node, Graph.number_of_nodes()) for source_node in source_nodes)
            ]
    elif case == 2 or case == 3:
        for node in Graph.nodes:
            Graph.nodes[node]['feature'] = [
                -1 if node in source_nodes else df.node_states[node],
                Graph.degree[node],
                min(nx.shortest_path_length(Graph, source=source_node).get(node, Graph.number_of_nodes()) for source_node in source_nodes)
            ]

    return Graph


def find_nodes_to_block(output, infected, blocked, k):
    output = output.detach().numpy()
    output = output.flatten()
    output = output.tolist()

    for i in range(len(output)):
        if i in infected or i in blocked:
            output[i] = -999

    # Get the indices of the top k values
    top_k_indices = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:k]

    return top_k_indices


def process_gnn(Graph, budget=1, case=1):
    # sequence_of_blocked_nodes = []
    trained_model = GCN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    trained_model.load_state_dict(torch.load(f"Models/model_case_{case}.pt"))

    previous_uninfected_nodes = None
    # cnt = 0

    while True:
        # cnt += 1
        # visualize_graph(Graph)
        node_features, edge_index, edge_weight = get_graph_features(Graph)
        number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)

        if uninfected_nodes == previous_uninfected_nodes:
            break
        previous_uninfected_nodes = uninfected_nodes

        model_output = trained_model(node_features, edge_index, edge_weight)

        if len(uninfected_nodes) <= budget:
            newly_blocked_nodes = uninfected_nodes
        else:
            newly_blocked_nodes = find_nodes_to_block(model_output, infected_nodes, blocked_nodes, budget)

        # print(number_of_nodes, newly_blocked_nodes, infected_nodes, blocked_nodes, uninfected_nodes)

        for node in newly_blocked_nodes:
            Graph.nodes[node]['feature'][0] = 1
            # sequence_of_blocked_nodes.append(node)

        Graph = update_graph_environment(copy.deepcopy(Graph))

    number_of_nodes, infected_nodes, _, _ = get_graph_properties(Graph)
    ir = len(infected_nodes) / number_of_nodes

    # print(number_of_nodes, cnt, sequence_of_blocked_nodes, infected_nodes)

    return ir


def process_random(Graph, budget=1):
    previous_uninfected_nodes = None
    # sequence_of_blocked_nodes = []

    cnt = 0
    while True:
        cnt += 1
        # visualize_graph(Graph)
        number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)

        if uninfected_nodes == previous_uninfected_nodes:
            break
        previous_uninfected_nodes = uninfected_nodes

        if len(uninfected_nodes) <= budget:
            blocked_nodes = uninfected_nodes
        else:
            blocked_nodes = np.random.choice(uninfected_nodes, budget, replace=False)

        for node in blocked_nodes:
            Graph.nodes[node]['feature'][0] = 1
            # sequence_of_blocked_nodes.append(node)

        Graph = update_graph_environment(copy.deepcopy(Graph))

    number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)
    ir = len(infected_nodes) / number_of_nodes

    # print(number_of_nodes, cnt, sequence_of_blocked_nodes, infected_nodes)

    return ir



def process_max_degree(Graph, budget=1):
    previous_uninfected_nodes = None

    while True:
        number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)

        if uninfected_nodes == previous_uninfected_nodes:
            break
        previous_uninfected_nodes = uninfected_nodes

        degree_dict = {}
        for node in Graph.nodes:
            if (Graph.nodes.data('feature')[node][0] < AUTHENTIC_OPINION
                    and Graph.nodes.data('feature')[node][0] > INFECTED_OPINION):
                degree_dict[node] = Graph.degree[node]

        if len(degree_dict) <= budget:
            blocked_nodes = uninfected_nodes
        else:
            blocked_nodes = sorted(degree_dict, key=lambda k: degree_dict[k], reverse=True)[:budget]

        for node in blocked_nodes:
            Graph.nodes[node]['feature'][0] = 1

        Graph = update_graph_environment(copy.deepcopy(Graph))

    number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)
    ir = len(infected_nodes) / number_of_nodes

    return ir


def process_max_degree_dynamic(Graph, budget=1):
    previous_uninfected_nodes = None

    while True:
        number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)

        if uninfected_nodes == previous_uninfected_nodes:
            break
        previous_uninfected_nodes = uninfected_nodes

        degree_dict = {}
        for node in Graph.nodes:
            if (Graph.nodes.data('feature')[node][0] < AUTHENTIC_OPINION
                    and Graph.nodes.data('feature')[node][0] > INFECTED_OPINION):
                degree_dict[node] = Graph.nodes[node]['feature'][1]

        if len(degree_dict) <= budget:
            blocked_nodes = uninfected_nodes
        else:
            blocked_nodes = sorted(degree_dict, key=lambda k: degree_dict[k], reverse=True)[:budget]

        for node in blocked_nodes:
            Graph.nodes[node]['feature'][0] = 1

        Graph = update_graph_environment(copy.deepcopy(Graph))

    number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(Graph)
    ir = len(infected_nodes) / number_of_nodes

    return ir


def comparison():
    csv_df = pd.DataFrame(columns=['case', 'budget', 'gnn', 'random', 'max_degree_static', 'max_degree_dynamic'])

    for case in range(1, 4):
        print("Case ", case)

        for i in range(0, 100):
            num_nodes = 50
            Graph, _, _, _, _ = generate_graph(num_nodes=num_nodes)

            for action_budget in range(1, 4):
                ir_gnn = process_gnn(copy.deepcopy(Graph), budget=action_budget, case=case)
                ir_random = process_random(copy.deepcopy(Graph), budget=action_budget)
                ir_max_degree = process_max_degree(copy.deepcopy(Graph), budget=action_budget)
                ir_max_degree_dynamic = process_max_degree_dynamic(copy.deepcopy(Graph), budget=action_budget)

                new_list = [case, action_budget, ir_gnn, ir_random, ir_max_degree, ir_max_degree_dynamic]
                csv_df.loc[len(csv_df)] = new_list

    grouped_df = csv_df.groupby(['budget']).agg('mean').reset_index()
    grouped_df.to_csv('./Comparison/comparison_modified.csv', index=False)

comparison()