from llm.prompt_template.a_node import *
from llm.prompt_template.a_pod import *
from llm.prompt_template.a_graph import GRAPH_MERGE_PROMPT


a_analysis_dict = {
    'fault_type_infer': {
        'node_cpu': {
            'metrics': NODE_CPU_METRICS,
            'prompt': NODE_CPU_PROMPT,
            'related_fault_types': ['Node CPU Failure', 'Node CPU Climb']
        },
        'node_memory': {
            'metrics': NODE_MEMORY_METRICS,
            'prompt': NODE_MEMORY_PROMPT,
            'related_fault_types': ['Node Memory Consumption']
        },
        'node_disk': {
            'metrics': NODE_DISK_METRICS,
            'prompt': NODE_DISK_PROMPT,
            'related_fault_types': ['Node Disk Read IO Consumption', 'Node Disk Write IO Consumption', 'Node Disk Space Consumption']
        },
        'pod_cpu': {
            'metrics': POD_CPU_METRICS,
            'prompt': POD_CPU_PROMPT,
            'related_fault_types': ['K8s CPU Load']
        },
        'pod_memory': {
            'metrics': POD_MEMORY_METRICS,
            'prompt': POD_MEMORY_PROMPT,
            'related_fault_types': ['K8s Memory Load']
        },
        'pod_disk': {
            'metrics': POD_DISK_METRICS,
            'prompt': POD_DISK_PROMPT,
            'related_fault_types': ['K8s Disk Read IO Load', 'K8s Disk Write IO Load']
        },
        'pod_process': {
            'metrics': POD_PROCESS_METRICS,
            'prompt': POD_PROCESS_PROMPT,
            'related_fault_types': ['K8s Process Terminated']
        },
        'pod_network': {
            'metrics': POD_NETWORK_METRICS,
            'prompt': POD_NETWORK_PROMPT,
            'related_fault_types': ['K8s Network Delay', 'K8s Network Packet Loss', 'K8s Network Packet Corruption', 'K8s Network Packet Duplication']
        }
    },
    'ent_induction': {},
    'graph_induction': {}
}

merge_template = {
    'node': NODE_MERGE_PROMPT,
    'pod': POD_MERGE_PROMPT
}

SUMMARIZATION_PROMPT = """
## Observations

{observations}

## Task

Summarize the fault type of each entity. The candidates are ['Node CPU Failure', 'Node CPU Climb', 'Node Memory Consumption', 'Node Disk Read IO Consumption', 'Node Disk Write IO Consumption', 'Node Disk Space Consumption', 'K8s CPU Load', 'K8s Memory Load', 'K8s Network Delay', 'K8s Network Packet Loss', 'K8s Network Packet Corruption', 'K8s Network Packet Duplication', 'K8s Disk Read IO Load', 'K8s Disk Write IO Load', 'K8s Process Terminated', 'No fault', 'Unknown'].
The response format MUST be in JSON format. The detailed format is: {detailed_format}
"""

