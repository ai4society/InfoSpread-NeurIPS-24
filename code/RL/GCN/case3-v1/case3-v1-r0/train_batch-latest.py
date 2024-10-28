import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from MisInfoSpread import MisInfoSpread
from experienceBuffer import ReplayBuffer
import json
import argparse
import time
from itertools import cycle, islice
import json

# Helper function for state processing
def flatten(state):
    return [val * i for val, adj in zip(state.node_states, state.adjacency_matrix) for i in adj]


def train(misinfo, policy, target, optimizer, criterion, memory, device, params_string, args):

    # Epsilon for epsilon-greedy action selection and other parameters
    eps = 1
    emax = args.episodes  # Number of episodes for training
    batch_size = args.batch_size
    total, T = 1, 5

    total_start_time = time.time()  # Start time for total training

    output_dict = {}

    mean_trust_dict = {}
    for e in range(emax):
        t = 1
        episode_start_time = time.time()  # Start time for the episode

        with open(f"./output/log_{params_string}.txt", "a") as log_file:
            log_file.write(f"\nStarting episode {e + 1} of {emax}\n")

        print(f"\nStarting episode {e + 1} of {emax}")
        states = misinfo.generate_states(args.states_per_episode)

        candidate_nodes_np = misinfo.find_neighbor_batch(states)

        while any(candidate_nodes for candidate_nodes in candidate_nodes_np):
            if random.uniform(0, 1) >= eps:
                policy.eval()
                blockernode_np = []
                for state, candidate_nodes in zip(states, candidate_nodes_np):
                    if candidate_nodes:
                        expectation_values = []
                        for cand_node in candidate_nodes:
                            temp_ns, _, _ = misinfo.step(copy.deepcopy(state), [cand_node])

                            # output_tensor = torch.FloatTensor(flatten(temp_ns)).view(1, -1).to(device)
                            temp_ns_node_features = torch.tensor(temp_ns.node_features, dtype=torch.float).to(device)
                            temp_ns_edge_index = torch.tensor(temp_ns.edge_index, dtype=torch.int64).t().contiguous().to(device)
                            temp_ns_edge_weight = torch.tensor(temp_ns.edge_weight, dtype=torch.float).to(device)

                            expected_infection = policy(temp_ns_node_features, temp_ns_edge_index, temp_ns_edge_weight)
                            expectation_values.append( (expected_infection, cand_node) )

                        # sort the expectation values based on the expected infection
                        expectation_values.sort(key=lambda x: x[0], reverse=True)

                        if len(expectation_values) < args.actions:
                            blockernode_np.append([node for _, node in expectation_values])
                        else:
                            blockernode_np.append([node for _, node in expectation_values[:args.actions]])

                    else:
                        blockernode_np.append([])
            else:
                blockernode_np = [random.sample(candidate_nodes, min(len(candidate_nodes), args.actions)) 
                                if candidate_nodes else [] for candidate_nodes in candidate_nodes_np]

            # print('------------------')
            # print('Candidate Nodes: ', candidate_nodes_np)
            # print('Blocker Selection: ', blockernode_np)

            policy.train()
            next_states, reward_np, done_np = misinfo.step_batch(copy.deepcopy(states), blockernode_np)
            candidate_nodes_np = misinfo.find_neighbor_batch(next_states)

            # print('States: ', [tmp.node_states for tmp in states])
            # print('Next States: ', [tmp.node_states for tmp in next_states])
            # print('Node Features: ', [tmp.node_features for tmp in next_states])
            # print([tmp.adjacency_matrix for tmp in next_states], end='\n\n')
            # print('Rewards: ', reward_np)
            # print([tmp.time_step for tmp in next_states], end='\n\n')
            # print('------------------')

            ########## New batch processing version ##########

            for state, blockernode, reward, next_state, done in zip(states, blockernode_np, reward_np, next_states, done_np):
                if len(blockernode) > 0:
                    memory.push(state, blockernode, reward, next_state, done)   


            if len(memory) >= batch_size:
                batch_data = memory.sample(batch_size)

                batch_next_states = [next_state for _, _, _, next_state, _ in batch_data]

                next_states_tensor = torch.FloatTensor([flatten(next_state) for _, _, _, next_state, _ in batch_data]).to(device)

                rewards = torch.tensor([reward for _, _, reward, _, _ in batch_data], dtype=torch.float).to(device)

                policy_outputs = []
                for ns in batch_next_states:
                    ns_node_features = torch.tensor(ns.node_features, dtype=torch.float).to(device)
                    ns_edge_index = torch.tensor(ns.edge_index, dtype=torch.int64).t().contiguous().to(device)
                    ns_edge_weight = torch.tensor(ns.edge_weight, dtype=torch.float).to(device)

                    policy_outputs.append(policy(ns_node_features, ns_edge_index, ns_edge_weight))
                policy_outputs = torch.stack(policy_outputs).to(device).squeeze()

                # print('Policy Outputs: ', policy_outputs)
                # Compute the target values
                tarVals = rewards.clone()
                done_tags = [i for _,_,_,_,i in batch_data]
                for i, done_tag in enumerate(done_tags):
                    if not done_tag:
                        new_cand_nodes = misinfo.find_neighbor(batch_next_states[i])
                        target_exp_values = []
                        for cand in new_cand_nodes:
                            temp_ns, _, _ = misinfo.step(copy.deepcopy(batch_next_states[i]), [cand])
                            # temp_ns_tensor = torch.FloatTensor(flatten(temp_ns)).view(1, -1).to(device)

                            temp_ns_node_features = torch.tensor(temp_ns.node_features, dtype=torch.float).to(device)
                            temp_ns_edge_index = torch.tensor(temp_ns.edge_index, dtype=torch.int64).t().contiguous().to(device)
                            temp_ns_edge_weight = torch.tensor(temp_ns.edge_weight, dtype=torch.float).to(device)

                            target_exp_value = target(temp_ns_node_features, temp_ns_edge_index, temp_ns_edge_weight).detach().view(-1)[0]
                            target_exp_values.append( (target_exp_value, cand) )

                        target_exp_values.sort(key=lambda x: x[0], reverse=True)
                        if len(target_exp_values) < args.actions:
                            target_cand_nodes = [node for _, node in target_exp_values]
                        else:
                            target_cand_nodes = [node for _, node in target_exp_values[:args.actions]]
                        
                        target_ns, _, _ = misinfo.step(copy.deepcopy(batch_next_states[i]), target_cand_nodes)
                        # target_ns_tensor = torch.FloatTensor(flatten(target_ns)).view(1, -1).to(device)
                        target_ns_node_features = torch.tensor(target_ns.node_features, dtype=torch.float).to(device)
                        target_ns_edge_index = torch.tensor(target_ns.edge_index, dtype=torch.int64).t().contiguous().to(device)
                        target_ns_edge_weight = torch.tensor(target_ns.edge_weight, dtype=torch.float).to(device)

                        tarVals[i] += target(target_ns_node_features, target_ns_edge_index, target_ns_edge_weight).detach().view(-1)[0]

                # print('Policy Outputs: ', policy_outputs)
                # print('Target Values: ', tarVals)
                # Compute loss
                loss = criterion(policy_outputs, tarVals)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print(f"Time step {t} of episode {e + 1} finished with loss: {loss.item() if 'loss' in locals() else 'N/A'}")
            
            if total % T == 0:
                target.load_state_dict(policy.state_dict())

            eps = max(0.1, 1 - 0.9 * ((e * t) / emax))

            states = next_states
            t += 1
            total += 1

            # Check if we should break out of the loop
            if all(done_np):
                break

        episode_end_time = time.time()  # End time for the episode
        episode_duration = round(episode_end_time - episode_start_time, 2)

        avg_reward = round(sum(reward_np) / len(reward_np), 4)

        with open(f"./output/log_{params_string}.txt", "a") as log_file:
            log_file.write(f"Episode {e + 1} finished with loss: {round(loss.item(), 4) if 'loss' in locals() else 'N/A'} | ran for {t} timesteps | final avg reward: {avg_reward} | took {episode_duration} seconds\n")
    
        output_dict[e+1] = {'loss': round(loss.item(), 4) if 'loss' in locals() else 'N/A', 'timesteps': t, 'average_reward': avg_reward, 'duration': episode_duration}
        print(f"Episode {e + 1} finished with loss: {round(loss.item(), 4)  if 'loss' in locals() else 'N/A'} | ran for {t} timesteps | final avg reward: {avg_reward} | took {episode_duration} seconds")

    # exit()
    total_end_time = time.time()  # End time for total training
    total_duration = total_end_time - total_start_time

    with open(f"./output/log_{params_string}.txt", "a") as log_file:
        log_file.write(f"Total training finished in {total_duration:.2f} seconds\n")
    
    print(f"Total training finished in {total_duration:.2f} seconds")

    # Save the models
    torch.save(policy.state_dict(), f"./saved_models/current_model_{params_string}.pt")
    torch.save(target.state_dict(), f"./saved_models/target_model_{params_string}.pt")

    return output_dict


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    misinfo = MisInfoSpread(num_nodes=args.max_nodes, max_time_steps=args.max_steps, 
                            trust_on_source=args.source_trust, count_infected_nodes=args.infected_nodes, 
                            count_actions=args.actions)

    policy = misinfo.get_nnet_model().to(device)
    target = misinfo.get_nnet_model().to(device)

    optimizer = optim.Adam(policy.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    memory = ReplayBuffer(1000000)

    params_string = f'{args.infected_nodes}_{args.actions}_mn{args.max_nodes}_ms{args.max_steps}_st{args.source_trust}'
    
    log_file = open(f"./output/log_{params_string}.txt", "w")
    log_file.write(f"####### Using device: {device} #######\n")
    log_file.close()

    output_dict = train(misinfo, policy, target, optimizer, criterion, memory, device, params_string, args)

    with open(f"./output/output_{params_string}.json", "w") as output_file:
        json.dump(output_dict, output_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL model to prevent misinformation spread")
    parser.add_argument("--max_nodes", type=int, default=10, help="Maximum number of nodes")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of timesteps")
    parser.add_argument("--source_trust", type=float, default=1, help="Trust value on source")
    parser.add_argument("--infected_nodes", type=int, default=1, help="Number of infected nodes")
    parser.add_argument("--actions", type=int, default=1, help="Number of actions to choose")

    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train for")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--states_per_episode", type=int, default=100, help="Number of states to generate per episode")

    args = parser.parse_args()
    main(args)