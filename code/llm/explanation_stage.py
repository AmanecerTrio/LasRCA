from llm.common import *
import os
import json
import numpy as np


with open(f'{prompt_base_path}/iteration-{k}/explanation/record.json') as f:
    record = json.load(f)


time_count = dict()
for i in range(len(record['sample_indices'])):
    start = time.time()

    print(f'{i}: start')
    sample_index, ent_index, prompt_type = record['sample_indices'][i], record['ent_indices'][i], record['prompt_types'][i]
    with open(f'{prompt_base_path}/explanation/{sample_index}/{ent_index}/{prompt_type}.txt') as r:
        prompt = r.read()
    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    save_dir = f'{response_base_path}/explanation/{sample_index}/{ent_index}'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{prompt_type}.txt', 'w') as w:
        w.write(chat_completion.choices[0].message.content)

    end = time.time()
    time_count[f'{sample_index}-{ent_index}-{prompt_type}'] = end - start
    print(f'{i}: end')

with open(f'{base_path}/llm/time_count.json', 'w') as f:
    json.dump(time_count, f, indent=2)

with open(f'{base_path}/llm/time_count.json') as f:
    time_count = json.load(f)
print(np.average(list(time_count.values())))
