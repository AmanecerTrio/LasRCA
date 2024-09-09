import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from tasks.util import generate_batch_edge_index


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout):
        super().__init__()
        self.conv1 = gnn.GATv2Conv(in_channels=in_channels,
                                   out_channels=out_channels,
                                   heads=heads,
                                   dropout=dropout,
                                   add_self_loops=False)
        self.conv2 = gnn.GATv2Conv(in_channels=out_channels * heads,
                                   out_channels=int(out_channels / heads),
                                   heads=heads,
                                   dropout=dropout,
                                   add_self_loops=False)

    def forward(self, x, edge_index):
        batch_size = x.shape[0]
        x = x.view(x.shape[0] * x.shape[1], x.shape[2]).contiguous()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = x.view(batch_size, int(x.shape[0] / batch_size), x.shape[1]).contiguous()
        return x


class FeatureFusion(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))  # 用于记录device.
        self.meta_data = meta_data
        self.GAT_net = GATNet(in_channels=param_dict['eff_in_dim'],
                              out_channels=param_dict['eff_GAT_out_channels'],
                              heads=param_dict['eff_GAT_heads'],
                              dropout=param_dict['eff_GAT_dropout'])
        self.linear_dict = nn.ModuleDict()
        for ent_type in self.meta_data['ent_types']:
            index_pair = self.meta_data['ent_fault_type_index'][ent_type]
            self.linear_dict[ent_type] = nn.Linear(param_dict['eff_GAT_out_channels'], index_pair[1] - index_pair[0])

    def forward(self, batch_data):
        x = batch_data['x_ent']
        ent_edge_index = batch_data['ent_edge_index'].to(self.device_marker).to(torch.int64)
        ent_edge_index = generate_batch_edge_index(ent_edge_index.shape[0], ent_edge_index, x.shape[1])
        x = self.GAT_net(x, ent_edge_index)
        return x
