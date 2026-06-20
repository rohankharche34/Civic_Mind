import networkx as nx
import numpy as np
from collections import defaultdict

G = nx.read_graphml("civicmind_graph.graphml")

print(f"No of edges: {G.number_of_edges()}")
print(f"No of nodes: {G.number_of_nodes()}")

#Creating the transition count

source_count = defaultdict(int)
transition_count = defaultdict(int)

MAX_GAP = 6

for u, v, data in G.edges(data = True):

    if data.get("edge_type") != "PRECEDED":
        continue

    if data.get("time_gap_hours", float("inf")) > MAX_GAP:
        continue 
    
    source = G.nodes[u].get("event_type")
    target = G.nodes[v].get("event_type")

    if source and target:

        transition_count[
            (source, target)
        ] += 1

        source_count[source] += 1

#calculating the cofidence and support
confidence_score = {}
    
for (source, target), count in transition_count.items():
    confidence = (count / source_count[source])
    confidence_score[
        (source, target)
    ] = round(confidence, 3)

support = {}

for edge, count in transition_count.items():
    support[edge] = count

for edge, count in support.items():
    print(f"{edge} -> {count}")

#Find the average and standard deviation
delay_store = defaultdict(list)

for u, v, data in G.edges(data = True):

    if data.get("edge_type") != "PRECEDED":
        continue

    if data.get("time_gap_hours", float("inf")) > MAX_GAP:
        continue

    source = G.nodes[u].get("event_type")
    target = G.nodes[v].get("event_type")
    delay = data.get("time_gap_hours")

    if delay is not None:
        delay_store[
            (source, target)
        ].append(float(delay))

#Find the average of delays
avg_delay = {}

for edge, delays in delay_store.items():
    avg_delay[edge] = round(sum(delays)/len(delays), 2)

std_delay = {}

for edge, delay in delay_store.items():
    std_delay[edge] = round(np.std(delay), 2)
