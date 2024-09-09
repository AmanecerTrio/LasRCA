from llama_index.core import PromptTemplate


POD_CPU_METRICS = ['kpi_container_cpu_usage_seconds', 'kpi_container_cpu_user_seconds', 'kpi_container_cpu_system_seconds', 'kpi_container_cpu_cfs_throttled_seconds', 'kpi_container_cpu_cfs_throttled_periods']
POD_CPU_PROMPT = PromptTemplate("""\
## Fault Description

### CPU Load

Features:
- All the metrics ['kpi_container_cpu_usage_seconds', 'kpi_container_cpu_user_seconds', 'kpi_container_cpu_cfs_throttled_seconds', 'kpi_container_cpu_cfs_throttled_periods'] show significant growth rather than random fluctuations.
- The metric 'kpi_container_cpu_system_seconds' may slightly increase compared with metrics ['kpi_container_cpu_usage_seconds', 'kpi_container_cpu_user_seconds', 'kpi_container_cpu_cfs_throttled_seconds', 'kpi_container_cpu_cfs_throttled_periods'].

{pod_cpu_load_examples}

## Question

{pod_cpu_test_examples}

Analysis hints:
- Fluctuations less than single digits can be ignored.
- If the metrics ['kpi_container_cpu_usage_seconds', 'kpi_container_cpu_user_seconds', 'kpi_container_cpu_cfs_throttled_seconds', 'kpi_container_cpu_cfs_throttled_periods'] do not show significant growth together, it is more likely to be no such fault.

Please address the following questions sequentially to methodically assess the situation. Let's think step by step.
1. Examine Fault Characteristics: Are there specific characteristics or patterns that align with known fault types? Pay extra attention to the locations where significant changes occur in the metrics. 
2. Determine the Duration of Fault Characteristics: How long do these characteristics persist? Calculate the number of data points or the time span over which the characteristics occur.
3. Fault Type Determination: Based on the identified characteristics and their durations, determine whether the Pod exhibits symptoms of the "CPU Load", or shows no such faults. Consider the descriptions of each fault type to guide your decision.
""")

POD_MEMORY_METRICS = ['kpi_container_memory_usage_MB', 'kpi_container_memory_working_set_MB', 'kpi_container_memory_rss']
POD_MEMORY_PROMPT = PromptTemplate("""\
## Fault Description

### Memory Load

Features:
- All the metrics ['kpi_container_memory_usage_MB', 'kpi_container_memory_working_set_MB', 'kpi_container_memory_rss'] show significant growth rather than random fluctuations. 

{pod_memory_load_examples}

## Question

{pod_memory_test_examples}

Please address the following questions sequentially to methodically assess the situation. Let's think step by step.
1. Examine Fault Characteristics: Are there specific characteristics or patterns that align with known fault types? Pay extra attention to the locations where significant changes occur in the metrics. 
2. Determine the Duration of Fault Characteristics: How long do these characteristics persist? Calculate the number of data points or the time span over which the characteristics occur.
3. Fault Type Determination: Based on the identified characteristics and their durations, determine whether the Pod exhibits symptoms of the "Memory Load", or shows no such faults. Consider the descriptions of each fault type to guide your decision.
""")

POD_DISK_METRICS = ['kpi_container_fs_reads_MB', 'kpi_container_fs_usage_MB', 'kpi_container_fs_writes_MB', 'kpi_container_threads']
POD_DISK_PROMPT = PromptTemplate("""\
## Fault Description

### Disk Read IO Load

Features:
- 'kpi_container_fs_reads_MB' shows the most significant increase among metrics ['kpi_container_fs_reads_MB', 'kpi_container_fs_usage_MB', 'kpi_container_fs_writes_MB'].
- 'kpi_container_fs_writes_MB', 'kpi_container_fs_usage_MB' may also show significant increase but not as large as 'kpi_container_fs_reads_MB'.
- 'kpi_container_threads' may undergo significant changes or remain unchanged. If the range of variation on the value is only 1-2 and other features match this fault type, then the fault type is more likely to be Process Kill.

{pod_disk_read_io_load_examples}

### Disk Write IO Load

Features:
- 'kpi_container_fs_writes_MB' or 'kpi_container_fs_usage_MB' shows the most significant increase among metrics ['kpi_container_fs_reads_MB', 'kpi_container_fs_usage_MB', 'kpi_container_fs_writes_MB'].
- 'kpi_container_fs_reads_MB' may also show significant increase but not as large as 'kpi_container_fs_writes_MB' or 'kpi_container_fs_usage_MB'.
- 'kpi_container_threads' may undergo significant changes or remain unchanged. If the range of variation on the value is only 1-2 and other features match this fault type, then the fault type is more likely to be "Process Kill".

{pod_disk_write_io_load_examples}

## Question

{pod_disk_test_examples}

Analysis hints:
- Mainly focus on the intensity and duration of changes in ['kpi_container_fs_reads_MB', 'kpi_container_fs_writes_MB', 'kpi_container_fs_usage_MB'].
- The range of variation on the value in 'kpi_container_threads' MUST NOT be 1-2. Otherwise, it is more likely to be "Process Kill".
- Slight increase or decrease (commonly < 1) in ['kpi_container_fs_reads_MB', 'kpi_container_fs_writes_MB', 'kpi_container_fs_usage_MB'] does not indicate potential faults.

Please address the following questions sequentially to methodically assess the situation. Let's think step by step.
1. Examine Fault Characteristics: Are there specific characteristics or patterns that align with known fault types? Pay extra attention to the locations where significant changes occur in the metrics. 
2. Determine the Duration of Fault Characteristics: How long do these characteristics persist? Calculate the number of data points or the time span over which the characteristics occur.
3. Fault Type Determination: Based on the identified characteristics and their durations, determine whether the Pod exhibits symptoms of  the "Disk Read IO Load", the "Disk Write IO Load", or shows no such faults. Consider the descriptions of each fault type to guide your decision.
""")

POD_PROCESS_METRICS = ['kpi_container_cpu_usage_seconds', 'kpi_container_cpu_user_seconds', 'kpi_container_cpu_system_seconds', 'kpi_container_cpu_cfs_throttled_seconds', 'kpi_container_cpu_cfs_throttled_periods', 'kpi_container_memory_cache', 'kpi_container_memory_mapped_file', 'kpi_container_memory_usage_MB', 'kpi_container_memory_working_set_MB', 'kpi_container_memory_rss', 'kpi_container_fs_inodes', 'kpi_container_fs_reads_MB', 'kpi_container_fs_usage_MB', 'kpi_container_fs_writes_MB', 'kpi_container_threads']
POD_PROCESS_PROMPT = PromptTemplate("""\
## Fault Description

### Process Terminated

Features:
- 'kpi_container_threads' MUST change for 1-3 times and eventually tends to stabilize. The range on the value is only 1-2.
- Many other metrics show the same trend as 'kpi_container_threads,' that is, they fluctuate to some extent, but stabilize as 'kpi_container_threads' stabilizes.

{pod_process_terminated_examples}

## Question

{pod_process_test_examples}

Analysis hints:
- Mainly focus on the changes in 'kpi_container_threads'. The range of the value must be only 1-2 if there exists such fault.
- Many other metrics must show the same trend of fluctuations as the change of 'kpi_container_threads', and they have the similar fault durations. . Otherwise, there may be no such fault.

Please address the following questions sequentially to methodically assess the situation. Let's think step by step.
1. Examine Fault Characteristics: Are there specific characteristics or patterns that align with known fault types? Pay extra attention to the locations where significant changes occur in the metrics. 
2. Determine the Duration of Fault Characteristics: How long do these characteristics persist? Calculate the number of data points or the time span over which the characteristics occur.
3. Fault Type Determination: Based on the identified characteristics and their durations, determine whether the Pod exhibits symptoms of  the "Process Terminated", or shows no such faults. Consider the descriptions of each fault type to guide your decision.
""")

POD_NETWORK_METRICS = ['kpi_container_network_receive_packets', 'kpi_container_network_receive_MB', 'kpi_container_network_transmit_packets', 'kpi_container_network_transmit_MB', '<intensity>; type: upstream', '<intensity>; type: current', '<intensity>; type: downstream', '<duration>; type: upstream', '<duration>; type: current', '<duration>; type: downstream']
POD_NETWORK_PROMPT = PromptTemplate("""\
## Metric Description

Metrics whose names contain '<intensity>' or '<duration>' are extracted from the distributed traces. Here are the meanings of '<intensity>' and '<duration>':
- '<intensity>': the number of spans in the given span set in a minute.
- '<duration>': the average duration of the spans in the span set in a minute.
There are also three types denoting the relations between spans and the Pod:
- 'type: upstream': the target span set contains all parent spans of the spans executed in the Pod.
- 'type: current': the target span set contains all spans executed in the Pod.
- 'type: downstream': the target span set contains all child spans of spans executed in the Pod.
Here are some tips for using such metrics:
- If metrics with 'type: upstream' or 'type: current' have significant changes, the target Pod may suffer from faults, either happened in the Pod or its callee.
- If metrics with 'type: downstream' have significant changes, the faults may happen in the callee of the Pod (sometimes the Pod also is its callee itself).
- You should only consider Duration metric changes as indicative of a problem if they increase by an order of magnitude in the tens to hundreds (usually on the order of ten thousand). Smaller changes are normal and should not be a cause for concern.

## Fault Description

### Network Delay

Features:
- The most significant and abrupt increase in Duration metrics, commonly by an order of magnitude in hundreds or thousands.
- Potential slightly and inconsistent fluctuation: There may be irregular small fluctuations in metrics about packets counts, but there is no significant upward or downward trend consistent with the abnormal growth time of duration.

{pod_network_delay_examples}

### Network Packet Loss

Features:
- Lower intensity and increasing duration metrics: Intensity metrics with '<intensity>' may decrease due to fewer packets being processed. Metric '<duration>; type: upstream' have an obvious increase.
- Consistent decrease of packet counts: Look for a significant decrease in metrics ['kpi_container_network_receive_packets', 'kpi_container_network_transmit_packets'] whose trend is consistent with the abnormal growth time of duration.
- Lower change in '<duration>; type: current': Changes in '<duration>; type: current' are significantly weaker than those in '<duration>; type: upstream'. Moreover, when changes in '<duration>; type: upstream', '<duration>; type: current', and '<duration>; type: downstream' are exceptionally high, the changes in '<duration>; type: current' are weaker than those in '<duration>; type: downstream', which is also indicative of a Network Packet Loss fault.

{pod_network_packet_loss_examples}

### K8s Network Packet Corruption

Features:
- Lower intensity and increasing duration metrics: Intensity metrics with '<intensity>' may decrease due to fewer packets being processed. Metric '<duration>; type: upstream' have an obvious increase.
- Consistent decrease of packet counts: Look for a significant decrease in metrics ['kpi_container_network_receive_packets', 'kpi_container_network_transmit_packets'] whose trend is consistent with the abnormal growth time of duration.
- Higher change in '<duration>; type: current': Changes in '<duration>; type: current' are similar as those in '<duration>; type: upstream'.

{pod_network_packet_corruption_examples}

### Network Packet Duplication

Features:
- Apparent Packet Fluctuation: Packet duplication is characterized by an apparent fluctuation in network traffic. This can be detected in metrics like 'kpi_container_network_receive_packets' and 'kpi_container_network_transmit_packets', where the counts might be unusually high relative to the expected network load.
- Slightly Intensity and Duration Metrics Fluctuation: The <intensity> and <duration> metrics might not show an abnormal change, with fluctuations of tens or hundreds in original Duration metric values can be ignored.

{pod_network_packet_duplication_examples}

## Question

{pod_network_test_examples}

Analysis hints:
- Delay: Look for a significant increase in Duration metrics. Other metrics may have fluctuations.
- Packet Loss: Look for a significant, consistent decrease in packet counts and a significant increase in Duration and Intensity metrics. Note the lower change in '<duration>; type: current'.
- Packet Corruption: Look for significant, consistent changes in packet counts and significant changes in Duration and Intensity metrics. Note the higher change in '<duration>; type: current'.
- Packet Duplication: There are apparent fluctuations in metrics ['kpi_container_network_receive_packets', 'kpi_container_network_receive_MB', 'kpi_container_network_transmit_packets', 'kpi_container_network_transmit_MB']. Besides, Intensity and Duration metrics have no obvious changes.
- You should only consider <duration> metric changes as indicative of a problem if they increase by an order of magnitude in the tens to hundreds. Smaller changes are normal and should not be a cause for concern.

Please address the following questions sequentially to methodically assess the situation. Let's think step by step.
1. Examine Fault Characteristics: Are there specific characteristics or patterns that align with known fault types? Pay extra attention to the locations where significant changes occur in the metrics. 
2. Determine the Duration of Fault Characteristics: How long do these characteristics persist? Calculate the number of data points or the time span over which the characteristics occur.
3. Fault Type Determination: Based on the identified characteristics and their durations, determine whether the Pod exhibits symptoms of the "Network Delay", "Network Packet Loss", "Network Packet Corruption", "Network Packet Duplication", or shows no such faults. Consider the descriptions of each fault type to guide your decision.
""")

POD_MERGE_PROMPT = PromptTemplate("""\
## Observations

### CPU

{cpu}

### Disk

{disk}

### Memory

{memory}

### Network

{network}

### Process

{process}

## Question

Analysis hints:
- The fault in CPU, Disk and Process may result in significant changes in Memory.
- The fault in Disk may result in significant changes in CPU.
- The fault in Process may result in fluctuations in all observations.
- If fault characteristics exhibit in both Network and other observations (any of CPU, Disk, Memory, and Process), consider other observations rather than Network since Network has many fluctuations.

Address the following questions sequentially based on the observations. Let's think step by step.
1. What is the most likely duration of the fault? Consider mutations that occur at a specific point. This is likely the initial point where the fault occurred. Do not focus on the whole time.
2. Which fault characteristic is most significant and accompanies the entire period of time?
3. What is the most probable fault type? Or there is 'No fault'? The candidates are 'K8s CPU Load', 'K8s Memory Load', 'K8s Network Delay', 'K8s Network Packet Loss', 'K8s Network Packet Corruption', 'K8s Network Packet Duplication', 'K8s Disk Read IO Load', 'K8s Disk Write IO Load', 'K8s Process Terminated' or 'No fault'. You MUST select the most probable candidate only, or clarify there is 'No fault'.
""")

