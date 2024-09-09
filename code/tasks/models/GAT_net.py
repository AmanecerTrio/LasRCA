import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
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


class GraphNet(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))  # 用于记录device.
        self.meta_data = meta_data
        self.merge_instance = nn.Linear(param_dict['window_size'], 1)
        self.GAT_net = GATNet(in_channels=param_dict['in_dim'],
                              out_channels=param_dict['GAT_out_channels'],
                              heads=param_dict['GAT_heads'],
                              dropout=param_dict['GAT_dropout'])
        self.linear_dict = nn.ModuleDict()
        for ent_type in self.meta_data['ent_types']:
            index_pair = self.meta_data['ent_fault_type_index'][ent_type]
            # self.linear_dict[ent_type] = nn.Linear(param_dict['GAT_out_channels'], index_pair[1] - index_pair[0])
            self.linear_dict[ent_type] = nn.Linear(param_dict['in_dim'], index_pair[1] - index_pair[0])

    def forward(self, batch_data):
        x = batch_data['embeddings'].to(self.device_marker)
        x = self.merge_instance(x.transpose(2, 3)).squeeze(dim=-1)
        ent_edge_index = batch_data['ent_edge_index'].to(self.device_marker).to(torch.int64)
        ent_edge_index = generate_batch_edge_index(ent_edge_index.shape[0], ent_edge_index, x.shape[1])
        # x = self.GAT_net(x, ent_edge_index)
        output = dict()
        for ent_type in self.meta_data['ent_types']:
            temp = x[:, self.meta_data['ent_type_index'][ent_type][0]:self.meta_data['ent_type_index'][ent_type][1], :]
            temp = temp.reshape(temp.shape[0] * temp.shape[1], temp.shape[2]).contiguous()
            output[ent_type] = self.linear_dict[ent_type](temp)
        return output
