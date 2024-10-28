import torch
import numpy as np
import pickle as pkl
from glob import glob
import os
import argparse
import regex as re
import pandas as pd
import statistics
import copy
import networkx as nx

from MisInfoSpread import MisInfoSpread
from MisInfoSpread import MisInfoSpreadState

def flatten(state):
    return [val * i for val, adj in zip(state.node_states, state.adjacency_matrix) for i in adj]

def get_node_features(state, num_nodes):

    infected_nodes = [i for i, node in enumerate(state.node_states) if node == -1]
    G = nx.from_numpy_array(state.adjacency_matrix)

    node_features = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        node_features[i].append( state.node_states[i] )
        neighbours = [item for item in G.neighbors(i) if item not in infected_nodes]
        node_features[i].append( len(neighbours) )
        min_val = 9999
        for node_index in infected_nodes:
            try:
                min_val = min(min_val, nx.shortest_path_length(G, i, node_index))
            except nx.NetworkXNoPath:
                pass

        node_features[i].append( min_val )

    edge_index = list(G.edges())
    edge_weight = [state.adjacency_matrix[i][j] for i, j in edge_index]

    return node_features, edge_index, edge_weight
    

def run_inference( dataset_path, model_path, nodes, max_steps, st, count_inf, count_actions):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    misinfo = MisInfoSpread(num_nodes=nodes, max_time_steps=max_steps, 
                        trust_on_source=st, count_infected_nodes=count_inf, 
                        count_actions=count_actions)

    model = misinfo.get_nnet_model().to(device)
    model.load_state_dict(torch.load(model_path , map_location=torch.device(device)))

    states = pkl.load(open(dataset_path, 'rb'))
    for state in states:
        # state.node_states = [int(x) for x in state.node_states]
        for i in range(len(state.adjacency_matrix)):
            for j in range(len(state.adjacency_matrix[i])):
                if state.adjacency_matrix[i][j] != 0:
                    state.adjacency_matrix[i][j] = 1
                else:
                    state.adjacency_matrix[i][j] = 0
        
        state.node_features, state.edge_index, state.edge_weight = get_node_features(state, nodes)

        state = MisInfoSpreadState(state.node_states, state.adjacency_matrix, state.node_features, state.edge_index, state.edge_weight, 0)

    candidate_nodes = misinfo.find_neighbor_batch(states)

    while any(candidate_node for candidate_node in candidate_nodes):
        blockernode_np = []
        count = 0
        for state, cand_nodes in zip(states, candidate_nodes):
            print("Processing states ", count, end='\r')
            if cand_nodes:
                expectation_values = []
                for cand_node in cand_nodes:
                    temp_ns, _, _ = misinfo.step(copy.deepcopy(state), [cand_node])

                    # output_tensor = torch.FloatTensor(flatten(temp_ns)).view(1, -1).to(device)
                    temp_ns_node_features = torch.tensor(temp_ns.node_features, dtype=torch.float).to(device)
                    temp_ns_edge_index = torch.tensor(temp_ns.edge_index, dtype=torch.int64).t().contiguous().to(device)
                    temp_ns_edge_weight = torch.tensor(temp_ns.edge_weight, dtype=torch.float).to(device)

                    expected_infection = model(temp_ns_node_features, temp_ns_edge_index, temp_ns_edge_weight)
                    expectation_values.append( (expected_infection, cand_node) )

                # sort the expectation values based on the expected infection
                expectation_values.sort(key=lambda x: x[0], reverse=True)

                if len(expectation_values) < count_actions:
                    blockernode_np.append([node for _, node in expectation_values])
                else:
                    blockernode_np.append([node for _, node in expectation_values[:count_actions]])

            else:
                blockernode_np.append([])

            count += 1
        next_states, rewards, done = misinfo.step_batch(states, blockernode_np)
        states = next_states
        candidate_nodes = misinfo.find_neighbor_batch(states)
        # print count of dones
        print("Done: ", done.count(True), " "*20)
        if all(done):
            break
    
    inf_rate = []
    for state in states:
        inf_rate.append(state.node_states.count(-1.0)/len(state.node_states))

    mean = round(statistics.mean(inf_rate), 4)
    std_dev = round(statistics.stdev(inf_rate), 4)
    print(f"Mean: {mean}, Std Dev: {std_dev}")
    return mean, std_dev

if __name__ == "__main__":
    datasets_path = '/work/bharath/InfoSpread-new/dataset/'
    models_path = 'saved_models/'

    # from argparge, get user input from --model and --dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=100, help='input node parameter 10/25/50')

    args = parser.parse_args()

    df = pd.DataFrame(columns=['model', 'd1_1', 'd1_2', 'd1_3', 'd2_1', 'd2_2', 'd2_3', 'd3_1', 'd3_2', 'd3_3'])

    for model in sorted(glob(models_path + '*.pt')):
        values = []
        if 'target' in model and 'mn'+str(args.nodes) in model:
            num_nodes = re.findall(r'mn(\d+)', model)
            source_trust = re.findall(r'st(\d+\.\d+)', model)
            infected = model.split('model')[-1].split('_')[1]
            actions = model.split('model')[-1].split('_')[2]

            values.append('{num_nodes}_{source_trust}_{infected}_{actions}'.format(num_nodes=num_nodes[0], source_trust=source_trust[0], infected=infected, actions=actions))
            for dataset_path in sorted(glob(datasets_path + '*.pkl')):
                if str(args.nodes) in dataset_path:
                    
                    for action in range(1, 4):
                        nodes = args.nodes
                        max_steps = 200
                        st = float(source_trust[0])
                        count_inf = os.path.basename(dataset_path).split('_')[-1].split('.')[0]
                        count_actions = action
                        print(f"Model: {os.path.basename(model)}, Dataset: {os.path.basename(dataset_path)}, Nodes: {nodes}, Max Steps: {max_steps}, ST: {st}, Count Inf: {count_inf}, Count Actions: {count_actions}")

                        mean, std_dev = run_inference(dataset_path, model, nodes, max_steps, st, count_inf, count_actions)
                        print('\n')
                        values.append('Mean: ' + str(mean) + ', Std Dev: ' + str(std_dev))

            df = df.append(pd.Series(values, index=df.columns), ignore_index=True)

    df.to_csv(f'output/inference_results_{args.nodes}.csv', index=False)