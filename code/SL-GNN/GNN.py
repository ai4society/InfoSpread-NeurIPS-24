import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

torch.manual_seed(42)

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.layer1 = GCNConv(input_size, hidden_size)
        self.layer2 = GCNConv(hidden_size, hidden_size)
        self.layer3 = GCNConv(hidden_size, num_classes)
        self.layer4 = nn.Linear(num_classes, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_features, edge_index, edge_weight=None):
        output = self.layer1(node_features, edge_index, edge_weight)
        output = torch.relu(output)
        output = self.layer2(output, edge_index, edge_weight)
        output = torch.relu(output)
        output = self.layer2(output, edge_index, edge_weight)
        output = torch.relu(output)
        output = self.layer2(output, edge_index, edge_weight)
        output = torch.relu(output)
        output = self.layer2(output, edge_index, edge_weight)
        output = torch.relu(output)
        output = self.layer3(output, edge_index, edge_weight)
        output = self.layer4(output)
        output = self.sigmoid(output)

        return output