from common import *
from prompt_template.induction import GRAPH_MERGE_PROMPT


for sample in selected_samples:
    sample_index = sample['sample_index']
    ent_info_list, ent_dict = [], dict()
    for i in range(len(sample['ent_indices'])):
        for ent_type, ent_pair in ent_pair_dict.items():
            if ent_pair[0] <= sample['ent_indices'][i] < ent_pair[1]:
                if ent_type not in ent_dict.keys():
                    ent_dict[ent_type] = []
                ent_dict[ent_type].append(sample["ent_names"][i])
                if ent_type == 'node' or ent_type == 'pod':
                    sample_index = sample['sample_index']
                    with open(f'{response_base_path}/stage-2/{sample_index}/{sample["ent_indices"][i]}.txt') as f:
                        content = f.read()
                    ent_info_list.append(f"### {sample['ent_names'][i]}\n\n{content}")
                break
    entities_list = '\n'.join([f'{ent_type}: {value}' for ent_type, value in ent_dict.items()])

    ent_info = '\n\n'.join(ent_info_list)
    graph_prompt = GRAPH_MERGE_PROMPT.format(
        observations='\n\n'.join(ent_info_list),
        entities='\n'.join([f'{ent_type}: {value}' for ent_type, value in ent_dict.items()]),
        relations='\n'.join([f'{sample["ent_edges"][0][index]} - {sample["ent_edges"][1][index]}' for index in range(len(sample["ent_edges"][0]))]),
    )
    save_dir = f'{prompt_base_path}/stage-3'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{sample_index}.txt', 'w') as w:
        w.write(graph_prompt)

gpt_query_list = []
for sample in selected_samples:
    sample_index = sample['sample_index']
    with open(f'./prompts/stage-3/{sample_index}.txt') as r:
        content = r.read()
    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": content}]
    )
    save_dir = f'{response_base_path}/stage-3'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{sample_index}.txt', 'w') as w:
        w.write(chat_completion.choices[0].message.content)
