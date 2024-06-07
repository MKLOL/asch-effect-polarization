import networkx as nx
import json
import pickle
import random


def getTweetData(opinion_pkl):
    G = nx.Graph()
    with open('graph_descriptions/final.edgelist', 'r') as file:
        for line in file:
            # Assuming each line is "NodeID1 NodeID2 Value" and they are separated by space
            node1, node2, _ = line.split()[:3]
            G.add_edge(node1, node2)

    # Load the pickle file
    with open(opinion_pkl, 'rb') as pickle_file:
        vax_gpt_labels = pickle.load(pickle_file)

    # Read and load the JSON file
    with open('graph_descriptions/vax_final.json', 'r') as json_file:
        vax_data = json.load(json_file)

    node_to_average = {}
    node_to_count = {}

    for node in G.nodes():
        # Check if the node exists in vax_data
        if node in vax_data:
            total_value = 0
            tweet_count = 0

            # Iterate through each tweet of the current node
            for tweet in vax_data[node]:
                tweet_id = tweet['tweet_id']  # Assuming each tweet has a 'tweet_id' field

                # Check if the tweet_id exists in vax_gpt_labels
                if tweet_id in vax_gpt_labels:
                    total_value += vax_gpt_labels[tweet_id]
                    tweet_count += 1

            # Calculate and print the average value if there are any tweets
            if tweet_count > 0:
                average_value = total_value / tweet_count
                node_to_average[node] = average_value
                node_to_count[node] = tweet_count
            else:
                node_to_average[node] = 0
                node_to_count[node] = 0

    opinion = []
    resistance = []

    # Iterate through the nodes
    for node in G.nodes():
        # Check if the count is zero
        if node_to_count[node] == 0:
            opinion.append(random.uniform(0.4, 0.6))
            resistance.append(random.uniform(0.2, 0.4))
        else:
            # Normalize the average to be between 0 and 1
            normalized_avg = min(max(node_to_average[node] / 10, 0), 1)
            opinion.append(normalized_avg)

            # Determine the resistance based on the count
            if 1 <= node_to_count[node] <= 5:
                resistance_value = random.uniform(0.4, 0.6)
            elif 5 <= node_to_count[node] <= 10:
                resistance_value = random.uniform(0.5, 0.7)
            elif 10 <= node_to_count[node] <= 20:
                resistance_value = random.uniform(0.6, 0.8)
            else:
                resistance_value = random.uniform(0.7, 0.9)

            resistance.append(resistance_value)
    return nx.convert_node_labels_to_integers(G, 0), resistance, opinion
