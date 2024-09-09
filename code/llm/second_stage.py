from common import *
from prompt_template.induction import merge_template
import time


for sample in selected_samples:
    for i in range(len(sample['ent_indices'])):
        for ent_type, ent_pair in ent_pair_dict.items():
            if ent_pair[0] <= sample['ent_indices'][i] < ent_pair[1]:
                if ent_type == 'node' or ent_type == 'pod':
                    sample_index = sample['sample_index']
                    observation_dict = dict()
                    for file in ent_files[ent_type]:
                        with open(f'{response_base_path}/stage-1/{sample_index}/{sample["ent_indices"][i]}/{ent_type}_{file}.txt') as f:
                            content = f.read()
                        observation_dict[file] = content
                    merge_prompt = merge_template[ent_type].format(**observation_dict)
                    save_dir = f'{prompt_base_path}/stage-2/{sample_index}'
                    os.makedirs(save_dir, exist_ok=True)
                    with open(f'{save_dir}/{sample["ent_indices"][i]}.txt', 'w') as w:
                        w.write(merge_prompt)
                break

gpt_query_list = []
for sample in selected_samples:
    for i in range(len(sample['ent_indices'])):
        for ent_type, ent_pair in ent_pair_dict.items():
            if ent_pair[0] <= sample['ent_indices'][i] < ent_pair[1]:
                if ent_type == 'node' or ent_type == 'pod':
                    sample_index = sample['sample_index']
                    with open(f'{prompt_base_path}/stage-2/{sample_index}/{sample["ent_indices"][i]}.txt') as f:
                        content = f.read()
                    chat_completion = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": content}]
                    )
                    save_dir = f'{response_base_path}/stage-2/{sample_index}'
                    os.makedirs(save_dir, exist_ok=True)
                    with open(f'{save_dir}/{sample["ent_indices"][i]}.txt', 'w') as w:
                        w.write(chat_completion.choices[0].message.content)
                break
