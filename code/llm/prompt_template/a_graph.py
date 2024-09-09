from llama_index.core import PromptTemplate


GRAPH_MERGE_PROMPT = PromptTemplate("""\
## Observations

{observations}

## Entities and Relations

### Entities

{entities}

### Relations

{relations}

## Question

RULES:
Each 'service' entity is associated with 4 'pods'. The names of these pods incorporate the service name followed by the suffixes '-0', '-1', '-2', or '2-0'. These pods must meet one of the following conditions:
- RULE 1: Only 1 pod displays fault characteristics. In this condition, the service MUST be classified to 'No Fault'.
- RULE 2: 3 or 4 pods display the same fault characteristics. In this condition, the service and ALL its associated 4 pods MUST be classified to this fault type.
- RULE 3: At most 1 fault type should be identified. If there exists two or more fault types in pods belonging to different services, this may be due to fault propagation between services. Therefore, the service associated with 4 pods that consistently exhibit specific faults should be classsified in the fault type. The other service and pods are 'No fault'.
- RULE 4: Once RULE 1 and RULE 2 are satisfied, all remaining entities are classified as 'No fault'.
- RULE 5: In the absence of satisfying RULE 4, if a service lacks the information of enough pods to make a determination, classify it as 'Unknown'.
You MUST obey the above rules. 

Address the following questions sequentially based on the observations. Let's think step by step.
1. What is the exact type of each entity? The candidates are ['Node CPU Failure', 'Node CPU Climb', 'Node Memory Consumption', 'Node Disk Read IO Consumption', 'Node Disk Write IO Consumption', 'Node Disk Space Consumption', 'K8s CPU Load', 'K8s Memory Load', 'K8s Network Delay', 'K8s Network Packet Loss', 'K8s Network Packet Corruption', 'K8s Network Packet Duplication', 'K8s Disk Read IO Load', 'K8s Disk Write IO Load', 'K8s Process Terminated', 'No fault', 'Unknown']. You MUST only select one candidate.
2. Explain the reasons for making judgments.
""")
