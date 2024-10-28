import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

torch.manual_seed(42)

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.layer1 = GCNConv(input_size, hidden_size)
        self.layer2 = GCNConv(hidden_size, hidden_size)
        self.layer3 = GCNConv(hidden_size, num_classes)
        self.layer4 = nn.Linear(num_classes, num_classes)
        self.global_pool = global_mean_pool
        self.final_layer = nn.Linear(1, 1)

    def forward(self, node_features, edge_index, edge_weight=None, batch=None):
        output = self.layer1(node_features, edge_index, edge_weight)
        output = torch.sigmoid(output)
        output = self.layer2(output, edge_index, edge_weight)
        output = torch.sigmoid(output)
        output = self.layer3(output, edge_index, edge_weight)
        output = self.layer4(output)

         # Pooling to get a single graph representation
        if batch is None:
            batch = torch.zeros(output.size(0), dtype=int, device=output.device)  # Assuming all nodes belong to the same graph
        pooled_output = self.global_pool(output, batch)
        
        # Final layer to get a single value for the entire graph
        graph_output = self.final_layer(pooled_output) 
        # graph_output = torch.sigmoid(graph_output)
        
        return graph_output