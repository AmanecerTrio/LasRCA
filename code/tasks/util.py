import os

from torch_sparse import SparseTensor
import numpy as np
import torch
import json
from shared_util.common import *
import torch.nn.functional as F
import copy

from llm.prompt_template.induction import a_analysis_dict
from llm.util import *


def generate_batch_edge_index(batch_size, edge_index, num_of_nodes):
    edge_index = edge_index.transpose(1, 2).contiguous()
    for i in range(batch_size):
        edge_index[i] += i * num_of_nodes
    edge_index = edge_index.view(edge_index.shape[0] * edge_index.shape[1], edge_index.shape[2]).t().contiguous()
    edge_index = SparseTensor(row=edge_index[0], col=edge_index[1])
    return edge_index


def label_to_multi_class_format(raw_y, num_of_fault_types=15):
    y = []
    raw_y = np.array(raw_y)
    for i in range(raw_y.shape[0]):
        y.append([])
        for j in range(raw_y.shape[1]):
            y[-1].append(np.zeros(num_of_fault_types))
            if raw_y[i][j] != 0:
                y[-1][-1][int(raw_y[i][j] - 1)] = 1
    return np.array(y)


def rearrange_y(meta_data, y, device):
    y_dict = dict()
    for ent_type in meta_data['ent_types']:
        index_pair = meta_data['ent_fault_type_index'][ent_type]
        temp = y[:, meta_data['ent_type_index'][ent_type][0]:meta_data['ent_type_index'][ent_type][1], index_pair[0]:index_pair[1]]
        y_dict[ent_type] = temp.reshape(temp.shape[0] * temp.shape[1], temp.shape[2]).contiguous().to(device)
    return y_dict


def linear_ramp_up(current, ramp_up_length):
    if ramp_up_length == 0:
        return 1.0
    else:
        current = np.clip(current / ramp_up_length, 0.0, 1.0)
        return float(current)


def mix_up(batch_data, y_dict, alpha, pseudo=False):
    mix_up_lambda = np.random.beta(alpha, alpha)
    mix_up_lambda = max(mix_up_lambda, 1 - mix_up_lambda)
    idx = torch.randperm(batch_data['ent_edge_index'].size(0))

    x_a, x_b = batch_data, {key: value[idx] for key, value in batch_data.items()}
    y_a, y_b = y_dict, {key: value[idx] for key, value in y_dict.items()}

    mix_x = {key: mix_up_lambda * x_a[key] + (1 - mix_up_lambda) * x_b[key] for key in batch_data.keys() if key != 'ent_edge_index'}
    mix_x['ent_edge_index'] = x_a['ent_edge_index']
    mix_y = {key: mix_up_lambda * y_a[key] + (1 - mix_up_lambda) * y_b[key] for key in y_dict.keys()} if not pseudo else y_a

    return mix_x, mix_y


def reverse_probs_to_graph(probs, ent_types, ent_type_index, ent_fault_type_index, fault_type_len):
    data_list = []
    for ent_type in ent_types:
        probs[ent_type] = np.array(probs[ent_type]).reshape(-1, ent_type_index[ent_type][1] - ent_type_index[ent_type][0], ent_fault_type_index[ent_type][1] - ent_fault_type_index[ent_type][0])
        if ent_fault_type_index[ent_type][0] > 0:
            prefix_labels = np.zeros((probs[ent_type].shape[0], probs[ent_type].shape[1], ent_fault_type_index[ent_type][0] - 0))
            probs[ent_type] = np.concatenate([prefix_labels, probs[ent_type]], axis=2)
        if ent_fault_type_index[ent_type][1] < fault_type_len:
            suffix_labels = np.zeros((probs[ent_type].shape[0], probs[ent_type].shape[1], fault_type_len - ent_fault_type_index[ent_type][1]))
            probs[ent_type] = np.concatenate([probs[ent_type], suffix_labels], axis=2)
        data_list.append(probs[ent_type])
    return np.concatenate(data_list, axis=1)


def get_fault_examples(dataset, probs, k=1):
    result_dict = dict()
    for label in [1]:  # [1, 0]
        index = np.where(dataset['data']['train']['y'] == label)

        fault_type_list = dataset['meta_data']['en_fault_type_list']
        for ent_type in dataset['meta_data']['ent_types']:
            ent_type_index = dataset['meta_data']['ent_type_index'][ent_type]
            ent_fault_type_index = dataset['meta_data']['ent_fault_type_index'][ent_type]
            valid = np.isin(index[1], np.array(range(ent_type_index[0], ent_type_index[1]))) & np.isin(index[2], np.array(range(ent_fault_type_index[0], ent_fault_type_index[1])))
            valid_index = (index[0][valid], index[1][valid], index[2][valid])
            if ent_type not in result_dict.keys():
                result_dict[ent_type] = dict()
            ent_fault_type_index = dataset['meta_data']['ent_fault_type_index'][ent_type]
            for i in range(ent_fault_type_index[0], ent_fault_type_index[1]):
                sample_indices, ent_indices = valid_index[0][valid_index[2] == i], valid_index[1][valid_index[2] == i]
                if fault_type_list[i] not in result_dict[ent_type].keys():
                    result_dict[ent_type][fault_type_list[i]] = {
                        'raw_data': [],
                        'data': [],
                        'prediction': [],
                        'confidence': [],
                        'label': []
                    }
                select_sample_num = k
                # select_sample_num = k if label == 1 else 2 * k
                for j in range(min(select_sample_num, sample_indices.shape[0])):
                    result_dict[ent_type][fault_type_list[i]]['raw_data'].append(dataset['data']['train']['raw_data'][sample_indices[j]][ent_indices[j]])
                    result_dict[ent_type][fault_type_list[i]]['data'].append(dataset['data']['train']['data'][sample_indices[j]][ent_indices[j]])
                    result_dict[ent_type][fault_type_list[i]]['label'].append(fault_type_list[i] if label == 1 else 'No ' + fault_type_list[i])
                    prob = probs[sample_indices[j]][ent_indices[j]][i]
                    if prob > 0.5:
                        result_dict[ent_type][fault_type_list[i]]['prediction'].append(fault_type_list[i])
                        result_dict[ent_type][fault_type_list[i]]['confidence'].append(prob)
                    else:
                        result_dict[ent_type][fault_type_list[i]]['prediction'].append('No ' + fault_type_list[i])
                        result_dict[ent_type][fault_type_list[i]]['confidence'].append(1 - prob)
    return result_dict


def binary_entropy(p):
    # Ensure numerical stability by clipping probabilities to avoid log(0)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def get_suspect_unlabeled_samples(dataset, probs, neighbor_hop=2, max_ent_per_graph=10, max_samples=100):
    samples = []
    fault_type_entropy = binary_entropy(probs)
    fault_type_entropy[np.array(dataset['data']['unlabeled']['y_mask']) == 1] = 0
    flatten_fault_type_entropy = fault_type_entropy.flatten()
    sorted_indices = np.argsort(flatten_fault_type_entropy)[::-1]
    index = np.unravel_index(sorted_indices, fault_type_entropy.shape)

    unique_value_set, final_index = set(), ...
    for i, value in enumerate(index[0]):
        unique_value_set.add(value)
        if len(unique_value_set) > max_samples:
            final_index = i
            break
    index = [index[0][:final_index], index[1][:final_index], index[2][:final_index]]

    fault_type_list = dataset['meta_data']['en_fault_type_list']
    for sample_index in np.unique(index[0]):
        ent_index_array = np.unique(index[1][index[0] == sample_index])
        ent_edge_index = np.array(dataset['data']['unlabeled']['ent_edge_index'][sample_index])
        ent_edge_index = ent_edge_index[:, ent_edge_index[0] != ent_edge_index[1]]
        related_ent_index_array, related_ent_edge_index_array = ent_index_array, None

        for hop in range(neighbor_hop):
            related_ent_edge_index_array = np.unique(np.sort(ent_edge_index[:, np.where(np.isin(ent_edge_index, related_ent_index_array))[1]], axis=0), axis=1)
            new_ent_index_array = np.unique(related_ent_edge_index_array)
            new_ent_index_array = np.flip(new_ent_index_array[~np.isin(new_ent_index_array, ent_index_array)])
            related_ent_index_array = np.concatenate((related_ent_index_array, new_ent_index_array))

        final_related_ent_index_list = []
        for i in related_ent_index_array:
            service_pair, pod_pair = dataset['meta_data']['ent_type_index']['service'], dataset['meta_data']['ent_type_index']['pod']
            if i not in final_related_ent_index_list and service_pair[0] <= i < service_pair[1]:
                final_related_ent_index_list.append(i)
                related_pods = np.unique(np.sort(ent_edge_index[:, np.where(np.isin(ent_edge_index, [i]))[1]], axis=0), axis=1)
                related_pods = np.intersect1d(related_pods[(related_pods >= pod_pair[0]) & (related_pods < pod_pair[1])], related_ent_index_array)
                final_related_ent_index_list.extend(related_pods.tolist())

        for i in related_ent_index_array:
            if i not in final_related_ent_index_list:
                final_related_ent_index_list.append(i)
        final_related_ent_index_array = np.array(final_related_ent_index_list)

        if final_related_ent_index_array.shape[0] > max_ent_per_graph:
            final_related_ent_index_array = final_related_ent_index_array[:max_ent_per_graph]
            related_ent_edge_index_array = np.unique(np.sort(related_ent_edge_index_array[:, np.where(np.isin(related_ent_edge_index_array, final_related_ent_index_array))[1]], axis=0), axis=1)
            related_ent_edge_index_array = related_ent_edge_index_array[:, np.isin(related_ent_edge_index_array[0], final_related_ent_index_array) & np.isin(related_ent_edge_index_array[0], final_related_ent_index_array)]
        samples.append({
            'sample_index': sample_index,
            'ent_indices': final_related_ent_index_array.tolist(),
            'ent_types': [],
            'ent_names': [dataset['meta_data']['ent_names'][j] for j in final_related_ent_index_array],
            'ent_edges': [[dataset['meta_data']['ent_names'][j] for j in related_ent_edge_index_array[0]], [dataset['meta_data']['ent_names'][j] for j in related_ent_edge_index_array[1]]],
            'raw_data': [dataset['data']['unlabeled']['raw_data'][sample_index][j] for j in final_related_ent_index_array],
            'data': [dataset['data']['unlabeled']['data'][sample_index][j] for j in final_related_ent_index_array],
            'prediction': [],
            'confidence': [],
            'label_names': [],
            'labels': []
        })

        for ent_index in final_related_ent_index_array:
            for ent_type, ent_type_index in dataset['meta_data']['ent_type_index'].items():
                if ent_type_index[0] <= ent_index < ent_type_index[1]:
                    samples[-1]['ent_types'].append(ent_type)
                    break
            samples[-1]['prediction'].append([])
            samples[-1]['confidence'].append([])
            for i in range(len(fault_type_list)):
                prob = probs[sample_index][ent_index][i]
                if prob > 0.5:
                    samples[-1]['prediction'][-1].append(fault_type_list[i])
                    samples[-1]['confidence'][-1].append(prob)
                else:
                    samples[-1]['prediction'][-1].append('No ' + fault_type_list[i])
                    samples[-1]['confidence'][-1].append(1 - prob)

            real_label = dataset['data']['unlabeled']['y'][sample_index][ent_index]
            if real_label.any():
                real_fault_type = fault_type_list[np.where(real_label == 1)[0][0]]
            else:
                real_fault_type = 'No fault'
            samples[-1]['label_names'].append(real_fault_type)
            samples[-1]['labels'].append(real_label)
    return samples


def get_fault_labeling_prompts(k):
    prompt_save_path = f"/workspace/project/working/2024/LasRCA/temp_data/A/llm/prompts/iteration-{k}/stage-1"
    os.makedirs(prompt_save_path, exist_ok=True)

    selected_samples = pkl_load(f'/workspace/project/working/2024/LasRCA/temp_data/A/selected_samples/selected_samples_k_{k}.pkl')
    fault_example_dict = pkl_load(f'{llm_content_save_path}/fault_examples.pkl')

    for sample in selected_samples:
        sample_index = sample['sample_index']
        sample_path = f'{prompt_save_path}/{sample_index}'
        for i in range(len(sample['ent_indices'])):
            example_dict = copy.deepcopy(fault_example_dict)
            ent_type = sample['ent_types'][i]
            if ent_type == 'node' or ent_type == 'pod':
                save_path = f'{sample_path}/{sample["ent_indices"][i]}'
                os.makedirs(save_path, exist_ok=True)
                for key in a_analysis_dict['fault_type_infer'].keys():
                    if ent_type not in key:
                        continue
                    observe_metrics = a_analysis_dict['fault_type_infer'][key]['metrics']
                    prompt_str = f'Given such metric values:\n'
                    prompt_str += metric_data_frame_to_str(sample['raw_data'][i], sample['data'][i], observe_metrics)
                    example_dict[ent_type][f'{key}_test_examples'] = prompt_str
                    prompt_template = a_analysis_dict['fault_type_infer'][key]['prompt'].format(**example_dict[ent_type])
                    with open(f'{save_path}/{key}.txt', 'w') as f:
                        f.write(prompt_template)


def get_explanation_prompts(dataset, test_probs, explanation_examples, k=0):
    record = {
        'sample_indices': [],
        'ent_indices': [],
        'prompt_types': []
    }
    index = np.where(test_probs > 0.5)
    for i in range(index[0].shape[0]):
        sample_index, ent_index, fault_type_index = index[0][i], index[1][i], index[2][i]
        ent_type = ''
        for key, value in dataset['meta_data']['ent_type_index'].items():
            if value[0] <= ent_index < value[1]:
                ent_type = key
                break
        if ent_type == 'service':
            continue

        save_dir = f"/workspace/project/working/2024/LasRCA/temp_data/A/llm/prompts/iteration-{k}/explanation/{sample_index}/{ent_index}"
        os.makedirs(save_dir, exist_ok=True)
        fault_type = dataset['meta_data']['en_fault_type_list'][index[2][i]]
        prompt_type, prompt_details = '', dict()
        for key, value in a_analysis_dict['fault_type_infer'].items():
            if fault_type in value['related_fault_types']:
                prompt_type, prompt_details = key, value
                break

        observe_metrics, prompt_template = prompt_details['metrics'], prompt_details['prompt']
        prompt_str = f'Given such metric values:\n'
        prompt_str += metric_data_frame_to_str(dataset['data']['test']['raw_data'][sample_index][ent_index], dataset['data']['test']['data'][sample_index][ent_index], observe_metrics)
        prompt_str += '\n\n'

        predict_list = []
        for related_fault_type in prompt_details['related_fault_types']:
            related_fault_type_index = dataset['meta_data']['en_fault_type_list'].index(related_fault_type)
            confidence = test_probs[sample_index][ent_index][related_fault_type_index]
            if confidence > 0.5:
                predict_list.append(f'Prediction from the fault type classification model: {related_fault_type} (Confidence: {format(confidence, ".3f")})')
            else:
                predict_list.append(f'Prediction from the fault type classification model: No {related_fault_type} (Confidence: {format(1 - confidence, ".3f")})')
        prompt_str += '\n'.join(predict_list)

        example_dict = copy.deepcopy(explanation_examples[ent_type])
        example_dict[f'{prompt_type}_test_examples'] = prompt_str
        final_prompt = prompt_template.format(**example_dict)
        with open(f'{save_dir}/{prompt_type}.txt', 'w') as f:
            f.write(final_prompt)
        record['sample_indices'].append(int(sample_index))
        record['ent_indices'].append(int(ent_index))
        record['prompt_types'].append(prompt_type)
    with open(f'/workspace/project/working/2024/LasRCA/temp_data/A/llm/prompts/iteration-{k}/explanation/record.json', 'w') as f:
        json.dump(record, f, indent=2)
