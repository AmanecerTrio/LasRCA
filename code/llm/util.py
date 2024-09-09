from llm.prompt_template.induction import a_analysis_dict
from shared_util.common import pkl_save, pkl_load
import os
import copy
import numpy as np


llm_content_save_path = "/workspace/project/working/2024/LasRCA/temp_data/A/llm"


def a_col_to_col_str(col):
    if '<intensity>' in col or '<duration>' in col:
        rearrange_list = col.split('; ')
        col_str = f'{rearrange_list[0]}; {rearrange_list[2]}'
    else:
        col_str = col.split('/', 1)[-1]
    return col_str


def metric_data_frame_to_str(raw_metric_df, z_scored_metric_df, observe_metrics):
    col_str_list = []
    for col in raw_metric_df.columns:
        col_str = a_col_to_col_str(col)
        if col_str in observe_metrics:
            if col_str == 'kpi_container_threads':
                diff_metric = np.diff(raw_metric_df[col]).tolist()
                final_data_list = [diff_metric[0]]
                final_data_list.extend(diff_metric)
                col_str_list.append(f"- '{col_str}': {[format(i, '.3f') for i in final_data_list]}")
            elif '<intensity>' in col_str or '<duration>' in col_str:
                raw_data_list, z_scored_data_list = raw_metric_df[col.replace('<duration>', '<intensity>')].tolist(), z_scored_metric_df[col].tolist()
                final_data_list = []
                for i in range(len(raw_data_list)):
                    if raw_data_list[i] != 0:
                        final_data_list.append(format(z_scored_data_list[i], '.3f'))
                    else:
                        final_data_list.append('NaN')
                col_str_list.append(f"- '{col_str}': {final_data_list}")
            else:
                col_str_list.append(f"- '{col_str}': {[format(i, '.3f') for i in z_scored_metric_df[col].tolist()]}")
    return '\n'.join(col_str_list)


def prepare_a_fault_type_infer_examples(fault_example_dict: dict):
    example_dict = dict()
    for ent_type, ent_fault_example_dict in fault_example_dict.items():
        if ent_type == 'node' or ent_type == 'pod':
            example_dict[ent_type] = dict()
            for fault_type, ent_fault_example_details in ent_fault_example_dict.items():
                example_name = f'{ent_type}_{fault_type.split(" ", 1)[-1].lower().replace(" ", "_")}_examples'
                observe_metrics = []
                for key in a_analysis_dict['fault_type_infer'].keys():
                    if key in example_name:
                        observe_metrics = a_analysis_dict['fault_type_infer'][key]['metrics']
                        break
                example_list = []
                for i in range(len(ent_fault_example_details['raw_data'])):
                    example_str = ''
                    example_str += f'Example{i + 1} metric values:\n'
                    example_str += metric_data_frame_to_str(fault_example_dict[ent_type][fault_type]['raw_data'][i], fault_example_dict[ent_type][fault_type]['data'][i], observe_metrics)
                    example_list.append(example_str)
                example_dict[ent_type][example_name] = '\n\n'.join(example_list)
    os.makedirs(llm_content_save_path, exist_ok=True)
    pkl_save(f'{llm_content_save_path}/fault_examples.pkl', example_dict)


def prepare_explanation_examples(fault_example_dict: dict):
    example_dict = dict()
    for ent_type, ent_fault_example_dict in fault_example_dict.items():
        if ent_type == 'node' or ent_type == 'pod':
            example_dict[ent_type] = dict()
            for fault_type, ent_fault_example_details in ent_fault_example_dict.items():
                example_name = f'{ent_type}_{fault_type.split(" ", 1)[-1].lower().replace(" ", "_")}_examples'
                observe_metrics = []
                for key in a_analysis_dict['fault_type_infer'].keys():
                    if key in example_name:
                        observe_metrics = a_analysis_dict['fault_type_infer'][key]['metrics']
                        break
                example_list = []
                for i in range(len(ent_fault_example_details['raw_data'])):
                    example_str = ''
                    example_str += f'Example{i + 1} metric values:\n'
                    example_str += metric_data_frame_to_str(fault_example_dict[ent_type][fault_type]['raw_data'][i], fault_example_dict[ent_type][fault_type]['data'][i], observe_metrics)
                    example_str += '\n\n'
                    example_str += f'Example{i + 1} prediction from the fault type classification model: {fault_example_dict[ent_type][fault_type]["prediction"][i]} (Confidence: {format(fault_example_dict[ent_type][fault_type]["confidence"][i], ".3f")})\n\n'
                    example_str += f'Example{i + 1} label: {fault_example_dict[ent_type][fault_type]["label"][i]}'
                    example_list.append(example_str)
                example_dict[ent_type][example_name] = '\n\n'.join(example_list)
    return example_dict
