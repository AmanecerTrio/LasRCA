import torch
import torch.nn as nn


class FaultClassifier(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data
        self.linear_dict = nn.ModuleDict()
        for ent_type in self.meta_data['ent_types']:
            index_pair = self.meta_data['ent_fault_type_index'][ent_type]
            self.linear_dict[ent_type] = nn.Linear(param_dict['eff_GAT_out_channels'], index_pair[1] - index_pair[0])

    def forward(self, x):
        output = dict()
        for ent_type in self.meta_data['ent_types']:
            temp = x[:, self.meta_data['ent_type_index'][ent_type][0]:self.meta_data['ent_type_index'][ent_type][1], :]
            temp = temp.reshape(temp.shape[0] * temp.shape[1], temp.shape[2]).contiguous()
            output[ent_type] = self.linear_dict[ent_type](temp)
        return output
