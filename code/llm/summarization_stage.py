from common import *
from prompt_template.induction import SUMMARIZATION_PROMPT
import time


for sample in selected_samples:
    sample_index = sample['sample_index']
    structure_output_format = dict()
    for i in range(len(sample['ent_indices'])):
        structure_output_format[sample["ent_names"][i]] = {
            'exact_fault_type': ''
        }
    with open(f'{response_base_path}/stage-3/{sample_index}.txt') as f:
        content = f.read()
    summarization_prompt = SUMMARIZATION_PROMPT.format(observations=content, detailed_format=str(structure_output_format))
    save_dir = f'{prompt_base_path}/summarization'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{sample_index}.txt', 'w') as w:
        w.write(summarization_prompt)

gpt_query_list = []
for sample in selected_samples:
    sample_index = sample['sample_index']
    with open(f'{prompt_base_path}/summarization/{sample_index}.txt') as r:
        content = r.read()
    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": content}]
    )
    save_dir = f'{response_base_path}/summarization'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{sample_index}.txt', 'w') as w:
        w.write(chat_completion.choices[0].message.content)
