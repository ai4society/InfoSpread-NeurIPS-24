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

from MisInfoSpread import MisInfoSpread
from MisInfoSpread import MisInfoSpreadState

def flatten(state):
    return [val * i for val, adj in zip(state.node_states, state.adjacency_matrix) for i in adj]

def run_inference( dataset_path, model_path, nodes, max_steps, st, count_inf, count_actions):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    misinfo = MisInfoSpread(num_nodes=nodes, max_time_steps=max_steps, 
                        trust_on_source=st, count_infected_nodes=count_inf, 
                        count_actions=count_actions)

    model = misinfo.get_nnet_model().to(device)
    model.load_state_dict(torch.load(model_path , map_location=torch.device(device)))

    states = pkl.load(open(dataset_path, 'rb'))
    # for state in states:
    #     state.node_states = [int(x) for x in state.node_states]
    #     for i in range(len(state.adjacency_matrix)):
    #         for j in range(len(state.adjacency_matrix[i])):
    #             if state.adjacency_matrix[i][j] != 0:
    #                 state.adjacency_matrix[i][j] = 1
    #             else:
    #                 state.adjacency_matrix[i][j] = 0

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
                    output_tensor = torch.FloatTensor(flatten(temp_ns)).view(1, -1).to(device)
                    expected_infection = model(output_tensor).detach().cpu().numpy()
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
    datasets_path = '/work/bharath/InfoSpread-new/deg_dataset/'
    models_path = 'saved_models/'

    # from argparge, get user input from --model and --dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=10, help='input node parameter 10/25/50')

    args = parser.parse_args()

    for dataset_path in sorted(glob(datasets_path + f'{args.nodes}/*.pkl')):
        df = pd.DataFrame(columns=['model', 'a1', 'a2', 'a3'])
        if str(args.nodes) in dataset_path:
            for model in sorted(glob(models_path + '*.pt')):
                values = []
                if 'target' in model and 'mn'+str(args.nodes) in model:
                    num_nodes = re.findall(r'mn(\d+)', model)
                    source_trust = re.findall(r'st(\d+\.\d+)', model)
                    infected = model.split('model')[-1].split('_')[1]
                    actions = model.split('model')[-1].split('_')[2]

                    values.append('{num_nodes}_{source_trust}_{infected}_{actions}'.format(num_nodes=num_nodes[0], source_trust=source_trust[0], infected=infected, actions=actions))
                            
                    for action in range(1, 4):
                        nodes = args.nodes
                        max_steps = 50
                        st = float(source_trust[0])
                        count_inf = os.path.basename(dataset_path).split('_')[-1].split('.')[0]
                        count_actions = action
                        print(f"Model: {os.path.basename(model)}, Dataset: {os.path.basename(dataset_path)}, Nodes: {nodes}, Max Steps: {max_steps}, ST: {st}, Count Inf: {count_inf}, Count Actions: {count_actions}")

                        mean, std_dev = run_inference(dataset_path, model, nodes, max_steps, st, count_inf, count_actions)
                        print('\n')
                        values.append('Mean: ' + str(mean) + ', Std Dev: ' + str(std_dev))

                    df = df.append(pd.Series(values, index=df.columns), ignore_index=True)

        dataset_name = os.path.basename(dataset_path).split('.')[0]
        # check if inference directory exists
        if not os.path.exists('output/inference'):
            os.makedirs('output/inference')
            print("Directory created for inference results")

        df.to_csv(f'output/inference/{dataset_name}_inf_res.csv', index=False)
        print(f"\n######## Results saved to output/inference/{dataset_name}_inf_res.csv ########\n\n")