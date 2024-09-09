import torch
import torch.nn as nn
from tasks.models.embed import PositionalEmbedding


class RepresentationLearning(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))  # 用于记录device.
        self.meta_data = meta_data
        self.different_modal_mapping_dict = nn.ModuleDict()
        self.positional_embedding_dict = nn.ModuleDict()
        self.modal_transformer_encoder_layer_dict = nn.ModuleDict()
        for modal_type in self.meta_data['modal_types']:
            num_of_o11y_features = 0
            for ent_type in self.meta_data['ent_types']:
                ent_index_pair = self.meta_data['ent_type_index'][ent_type]
                num_of_o11y_features += self.meta_data['ent_feature_num'][ent_type][modal_type] * (ent_index_pair[1] - ent_index_pair[0])

            self.positional_embedding_dict[modal_type] = PositionalEmbedding(in_features=param_dict['window_size'], num_of_o11y_features=num_of_o11y_features)
            self.different_modal_mapping_dict[modal_type] = nn.Linear(in_features=param_dict['window_size'], out_features=param_dict['orl_te_in_channels'])
            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=param_dict['orl_te_in_channels'], nhead=param_dict['orl_te_heads'])
            self.modal_transformer_encoder_layer_dict[modal_type] = nn.TransformerEncoder(transformer_encoder_layer, num_layers=param_dict['orl_te_layers'])

    def forward(self, batch_data):
        for modal_type in self.meta_data['modal_types']:
            batch_data[f'x_{modal_type}'] = batch_data[f'x_{modal_type}'].to(self.device_marker.device)
            batch_data[f'x_{modal_type}'] = self.positional_embedding_dict[modal_type](batch_data[f'x_{modal_type}']).contiguous()
            batch_data[f'x_{modal_type}'] = self.different_modal_mapping_dict[modal_type](batch_data[f'x_{modal_type}'])
            batch_data[f'x_{modal_type}'] = self.modal_transformer_encoder_layer_dict[modal_type](batch_data[f'x_{modal_type}'])
        return batch_data
