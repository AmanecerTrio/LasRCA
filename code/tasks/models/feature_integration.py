import torch
import torch.nn as nn


class FeatureIntegration(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data

        self.ent_transformer_encoder_layer_dict = nn.ModuleDict()
        self.ent_feature_align_dict = nn.ModuleDict()  # 用于实体对齐, 将各类别实体的特征映射到同一维度, 以输入GAT运算.

        in_dim = param_dict['efi_in_dim']

        self.ent_feature_length_dict = dict()
        for ent_type in self.meta_data['ent_types']:
            self.ent_feature_length_dict[ent_type] = sum([value for key, value in self.meta_data['ent_feature_num'][ent_type].items()])
            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=param_dict['efi_te_heads'])
            self.ent_transformer_encoder_layer_dict[ent_type] = nn.TransformerEncoder(transformer_encoder_layer, num_layers=param_dict['efi_te_layers'])
            self.ent_feature_align_dict[ent_type] = nn.Linear(self.ent_feature_length_dict[ent_type] * in_dim, param_dict['efi_out_dim'])

    def forward(self, batch_data):
        batch_size = batch_data['ent_edge_index'].shape[0]

        x_ent = []
        start_index = {modal_type: 0 for modal_type in self.meta_data['modal_types']}
        for ent_type in self.meta_data['ent_types']:
            for ent_index in range(self.meta_data['ent_type_index'][ent_type][0], self.meta_data['ent_type_index'][ent_type][1]):
                x = []
                for modal_type in self.meta_data['modal_types']:
                    modal_data = batch_data[f'x_{modal_type}'][:, start_index[modal_type]:start_index[modal_type] + self.meta_data['ent_feature_num'][ent_type][modal_type], :]
                    start_index[modal_type] += self.meta_data['ent_feature_num'][ent_type][modal_type]
                    x.append(modal_data)
                x = torch.cat(x, dim=1)
                x = self.ent_transformer_encoder_layer_dict[ent_type](x)
                x = x.view(batch_size, x.shape[1] * x.shape[2]).contiguous()
                x = self.ent_feature_align_dict[ent_type](x)
                x_ent.append(x)
        x_ent = torch.stack(x_ent, dim=1)
        batch_data['x_ent'] = x_ent
        return batch_data
