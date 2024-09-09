from common import *
import numpy as np
import json


def extract_json_blobs(content):
    str_index = 0
    while str_index < len(content):
        if content[str_index] == '{':
            for j in range(len(content) - 1, str_index, -1):
                if content[j] == '}':
                    try:
                        return json.loads(content[str_index:j+1])
                    except json.JSONDecodeError as e:
                        pass
        str_index += 1


dataset_path = f'{base_path}/metric_trace_log/dataset/narrow_few_shot_train_size_1'
all_dataset_path = f'{dataset_path}_k_{k}' if k != 0 else dataset_path

with open(f'{all_dataset_path}.pkl', 'rb') as pkl_reader:
    dataset = pickle.load(pkl_reader)

for sample in selected_samples:
    with open(f'{response_base_path}/summarization/{sample["sample_index"]}.txt') as f:
        llm_result = f.read().replace('\'', '\"')
    llm_result_dict = extract_json_blobs(llm_result)

    for ent_name, value in llm_result_dict.items():
        if ent_name in dataset['meta_data']['ent_names']:
            fault_type = value['exact_fault_type']
            if fault_type == "Node Disk Space IO Consumption":
                fault_type = "Node Disk Space Consumption"

            if isinstance(fault_type, list):
                continue
            ent_index = dataset['meta_data']['ent_names'].index(ent_name)
            fault_index, mask = 0, 0
            if fault_type.lower() == 'no fault':
                mask = 1
            if fault_type in dataset['meta_data']['en_fault_type_list']:
                fault_index = dataset['meta_data']['en_fault_type_list'].index(fault_type) + 1
                mask = 1
            dataset['data']['unlabeled']['y'][sample["sample_index"]][ent_index] = fault_index
            dataset['data']['unlabeled']['y_mask'][sample["sample_index"]][ent_index] = mask

with open(f'{dataset_path}_k_{k + 1}_GPT4.pkl', 'wb') as f:
    pickle.dump(dataset, f)

random_chosen_nums = [12, 24, 36, 48, 60]
for random_chosen_num in random_chosen_nums:
    with open(f'{all_dataset_path}.pkl', 'rb') as pkl_reader:
        dataset = pickle.load(pkl_reader)
    sample_indices = list(range(len(dataset['data']['unlabeled']['y_mask'])))
    chosen_list = np.random.choice(sample_indices, random_chosen_num, replace=False).tolist()
    for chosen_index in chosen_list:
        for i in range(len(dataset['meta_data']['ent_names'])):
            dataset['data']['unlabeled']['y_mask'][chosen_index][i] = 1
    with open(f'{dataset_path}_k_{k + 1}_random_labeling_{random_chosen_num}.pkl', 'wb') as f:
        pickle.dump(dataset, f)
