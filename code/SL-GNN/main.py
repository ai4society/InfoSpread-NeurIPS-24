import random
import torch
import torch.nn as nn
import torch.optim as optim
from GNN import GCN
from tqdm import tqdm
import copy

from graph_utils import (
    generate_graph,
    output_with_minimal_infection_rate,
    get_graph_properties,
    update_graph_environment,
    get_graph_features
)
from visualization_utils import visualize_loss, visualize_graph
from params import (
    EPISODE_MAX,
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,   
    LEARNING_RATE
)


if __name__ == '__main__':
    torch.manual_seed(42)

    for case in tqdm(range(1, 4)):
        print("Starting training for case ", case)

        # Initialize the policy
        policy = GCN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()

        total_training_loss_list = []  # List to store the loss values

        for graph_count in tqdm(range(1500)):
            node_count = 25
            Graph, node_features, edge_index, source_node, edge_weight = generate_graph(
                num_nodes=node_count,
            )

            _, prev_infected_nodes, prev_blocked_nodes, prev_uninfected_nodes = get_graph_properties(
                copy.deepcopy(Graph)
            )

            total_loss = 0
            iteration = 0

            for episode in range(EPISODE_MAX):
                node_features, edge_index, edge_weight = get_graph_features(copy.deepcopy(Graph))
                number_of_nodes, infected_nodes, blocked_nodes, uninfected_nodes = get_graph_properties(
                    copy.deepcopy(Graph)
                )

                if (episode != 0 and prev_infected_nodes == infected_nodes and
                    prev_blocked_nodes == blocked_nodes and prev_uninfected_nodes == uninfected_nodes):
                    break

                prev_infected_nodes = infected_nodes
                prev_uninfected_nodes = uninfected_nodes
                prev_blocked_nodes = blocked_nodes

                budget = random.randint(1, 3)
                target_output, new_blocked_nodes = output_with_minimal_infection_rate(
                    copy.deepcopy(Graph),
                    budget=budget
                )
                policy_output = policy(node_features, edge_index, edge_weight)

                policy.train()
                loss = criterion(policy_output, target_output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                for blocked_node in new_blocked_nodes:
                    Graph.nodes[blocked_node]['feature'][0] = 1
                Graph = update_graph_environment(copy.deepcopy(Graph))

                iteration += 1

            total_loss /= iteration
            total_training_loss_list.append(total_loss)

        visualize_loss(total_training_loss_list, case=case)

        print(f"\nTraining completed for case {case}!")
        print("Saving models...")
        current_model_path = f"Models/model_case_{case}.pt"
        torch.save(policy.state_dict(), current_model_path)
        print("Models saved successfully!")