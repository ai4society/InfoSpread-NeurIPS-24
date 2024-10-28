import random
import torch
from graph_utils import get_graph_properties
from params import INFECTION_THRESHOLD

def find_node_to_block(output, infected, blocked):
    """
    :param output: output of the model
    :param output:
    :param infected:
    :param blocked:
    :return: the most important node to block. node cannot be in infected or in blocked list
    """

    output = output.detach().numpy()
    output = output.flatten()
    output = output.tolist()

    for i in range(len(output)):
        if i in infected or i in blocked:
            output[i] = -1

    return output.index(max(output))


def get_greedy_model(graph):
    degree_dict = {}
    for node in graph.nodes:
        if graph.nodes.data('feature')[node][0] != -1 and graph.nodes.data('feature')[node][0] < INFECTION_THRESHOLD:
            degree_dict[node] = graph.degree[node]

    max_key = next(iter(degree_dict))
    for key in degree_dict:
        if degree_dict[key] > degree_dict[max_key]:
            max_key = key

    graph.nodes[max_key]['feature'][0] = -1

    return graph


def get_random_model(graph):
    remaining_nodes = []
    for node in graph.nodes:
        if graph.nodes[node]['feature'][0] != -1 and graph.nodes[node]['feature'][0] < INFECTION_THRESHOLD:
            remaining_nodes.append(node)

    blocked_node = random.choice(remaining_nodes)
    graph.nodes[blocked_node]['feature'][0] = -1

    return graph


def get_trained_model(model, graph):
    node_features = graph.nodes.data('feature')
    node_features = torch.tensor([node_feature[1] for node_feature in node_features], dtype=torch.float)
    edge_index = torch.tensor(list(graph.edges), dtype=torch.int64).t().contiguous()
    edge_weights = [graph[edge[0]][edge[1]]['weight'] for edge in graph.edges()]
    edge_weight = torch.tensor(edge_weights)
    model_output = model(node_features, edge_index, edge_weight)

    number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(graph)
    blocked_node = find_node_to_block(model_output, infected_nodes, blocked_nodes)
    graph.nodes[blocked_node]['feature'][0] = -1

    return graph

