import os
import pickle
from openai import OpenAI
import time


k = 0
client = OpenAI(api_key="your_api_key")


def find_strings_with_prefix(strings, prefix):
    return [s for s in strings if s.startswith(prefix)]


def generate_gpt_query(custom_id, content):
    return {
        "custom_id": f'{custom_id}',
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": content}],
            "max_tokens": 1000
        }
    }


base_path = '/workspace/project/working/2024/LasRCA/temp_data/A'
ent_count_dict = {'node': 0, 'service': 0, 'pod': 0}
ent_pair_dict = {'node': (0, 6), 'service': (6, 16), 'pod': (16, 56)}
ent_files = {'node': ['cpu', 'disk', 'memory'], 'pod': ['cpu', 'disk', 'memory', 'network', 'process']}
with open(f'{base_path}/selected_samples/selected_samples_k_{k}.pkl', 'rb') as pkl_reader:
    selected_samples = pickle.load(pkl_reader)
for sample in selected_samples:
    for i in range(len(sample['ent_indices'])):
        for ent_type, ent_pair in ent_pair_dict.items():
            if ent_pair[0] <= sample['ent_indices'][i] < ent_pair[1]:
                ent_count_dict[ent_type] += 1

prompt_base_path = f'{base_path}/llm/prompts/iteration-{k}'
os.makedirs(prompt_base_path, exist_ok=True)
response_base_path = f'{base_path}/llm/response/iteration-{k}'
os.makedirs(response_base_path, exist_ok=True)
