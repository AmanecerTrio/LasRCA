from llama_index.core import PromptTemplate


NODE_CPU_METRICS = ['system.cpu.pct_usage', 'system.cpu.user']
NODE_CPU_PROMPT = PromptTemplate("""\
## Fault Description

### CPU Failure

Features: Metrics ['system.cpu.pct_usage', 'system.cpu.user'] involves high growth over 1~2 points, then stabilizes or does not continue to rise significantly.

{node_cpu_failure_examples}

### CPU Climb

Features: Metrics ['system.cpu.pct_usage', 'system.cpu.user'] involves abnormal continuous growth over 3 or more points in their values, while there are no significant changes other metrics.

{node_cpu_climb_examples}

## Question

{node_cpu_test_examples}

Please address the following questions sequentially to methodically assess the situation. Let's think step by step.
1. Examine Fault Characteristics: Are there specific characteristics or patterns that align with known fault types? Pay extra attention to the locations where significant changes occur in the metrics. 
2. Determine the Duration of Fault Characteristics: How long do these characteristics persist? Calculate the number of data points or the time span over which the characteristics occur.
3. Fault Type Determination: Based on the identified characteristics and their durations, determine whether the node exhibits symptoms of the "CPU Failure", "CPU Climb", or shows no such faults. Consider the descriptions of each fault type to guide your decision.
""")

NODE_MEMORY_METRICS = ['system.mem.real.pct_useage', 'system.mem.pct_usage']
NODE_MEMORY_PROMPT = PromptTemplate("""\
## Fault Description

### Memory Consumption

Features: 
- Some metrics in ['system.mem.real.pct_useage', 'system.mem.pct_usage'] exhibit sudden increases and remain high for a period of time in their values, while there are no significant changes other metrics.
- The increase is obvious (commonly > 0.1).

{node_memory_consumption_examples}

## Question

{node_memory_test_examples}

Please address the following questions sequentially to methodically assess the situation. Let's think step by step.
1. Examine Fault Characteristics: Are there specific characteristics or patterns that align with known fault types?  Pay extra attention to the locations where significant changes occur in the metrics.
2. Determine the Duration of Fault Characteristics: How long do these characteristics persist? Calculate the number of data points or the time span over which the characteristics occur.
3. Fault Type Determination: Based on the identified characteristics and their durations, determine whether the node exhibits symptoms of the "Memory Consumption" or shows no such faults. Consider the descriptions of each fault type to guide your decision.
""")

NODE_DISK_METRICS = ['system.io.avg_q_sz', 'system.io.r_await', 'system.io.r_s', 'system.io.rkb_s', 'system.io.rkb_s', 'system.io.w_await', 'system.io.w_s']
NODE_DISK_PROMPT = PromptTemplate("""\
## Fault Description

### Disk Read IO Consumption

Features: 
- Substantial increases in metrics ['system.io.r_await', 'system.io.r_s', 'system.io.rkb_s'], which are directly related to the performance and efficiency of read operations.
- Other metrics may show minor fluctuations but the pronounced changes should predominantly be in the read metrics.

{node_disk_read_io_consumption_examples}

### Disk Write IO Consumption

Features: 
- Metrics ['system.io.avg_q_sz', 'system.io.w_await'] have obvious increases. The overall trend of anomalies in both ['system.io.avg_q_sz', 'system.io.w_await'] approach 10.

{node_disk_write_io_consumption_examples}

### Disk Space IO Consumption

Features: 
- Metrics ['system.io.avg_q_sz', 'system.io.w_await'] have significant increases. The overall trend of anomalies in both ['system.io.avg_q_sz', 'system.io.w_await'] approach 18.

{node_disk_space_consumption_examples}

## Question

{node_disk_test_examples}

Analysis hints:
- Check the scale and consistency of changes across different metrics. If read-related metrics are disproportionately higher than others, it points to a Disk Read IO Consumption fault.
- If there are high values in ['system.io.avg_q_sz', 'system.io.w_await'], consider the degree of abnormality. If most abnormal values are near 18, it suggests Disk Space IO Consumption. If the value is not too high which is around 10, it suggests Disk Write IO Consumption.

Please address the following questions sequentially to methodically assess the situation. Let's think step by step.
1. Examine Fault Characteristics: Are there specific characteristics or patterns that align with known fault types? Pay extra attention to the locations where significant changes occur in the metrics. 
2. Determine the Duration of Fault Characteristics: How long do these characteristics persist? Calculate the number of data points or the time span over which the characteristics occur.
3. Fault Type Determination: Based on the identified characteristics and their durations, determine whether the node exhibits symptoms of the "Disk Read IO Consumption", "Disk Write IO Consumption", "Disk Space IO Consumption", or shows no such faults. Consider the descriptions of each fault type to guide your decision.
""")

NODE_MERGE_PROMPT = PromptTemplate("""\
## Observations

### CPU

{cpu}

### Disk

{disk}

### Memory

{memory}

## Question

Address the following questions sequentially based on the observations. Let's think step by step.
1. What is the most likely duration of the fault?
2. Which fault characteristic is most significant and accompanies the entire period of time?
3. What is the most probable fault type? Or there is 'No fault'? The candidates are 'Node CPU Failure', 'Node CPU Climb', 'Node Memory Consumption', 'Node Disk Read IO Consumption', 'Node Disk Write IO Consumption', and 'Node Disk Space Consumption'.  You MUST select the most probable candidate only, or clarify there is 'No fault'.
""")
