import numpy as np
import pandas as pd
from collections import Counter
import math
from graphviz import Digraph


def load_data_and_describe(filepath):
    """
    Load data from a CSV file and print basic statistics.

    Args:
    - filepath: str, path to the CSV file

    Returns:
    - data: DataFrame, loaded data
    """
    data = pd.read_csv(filepath)
    
    print("--- DATA STATISTICS ---")
    print("DATA SHAPE: ")
    print(data.describe())
    print("----------------------------------------\n\n")
    
    return data 


def calculate_entropy(data):
    """
    Calculate entropy for a target column in the data.

    Args:
    - data: DataFrame, input data

    Returns:
    - entropy_val: float, entropy value
    """
    # Extract target column
    target_column = data.iloc[:, -1]
    
    # Count distinct values in the target column
    distinct_value_counts = Counter(target_column)
    
    total_samples = len(target_column)
    
    entropy_val = 0
    
    # Calculate entropy using the formula
    for count in distinct_value_counts.values():
        probability = count / total_samples
        entropy_val -= probability * math.log2(probability)
    
    return entropy_val


def calculate_average_entropy(data, attribute_name):
    """
    Calculate average entropy for a specific attribute.

    Args:
    - data: DataFrame, input data
    - attribute_name: str, attribute column name

    Returns:
    - avg_entropy: float, average entropy value
    """
    # Get unique values of the attribute
    unique_values = data[attribute_name].unique()
    
    total_samples = len(data)
    
    avg_entropy = 0

    # Calculate weighted entropy for each unique attribute value
    for value in unique_values:
        subset_data = data[data[attribute_name] == value]
        weight = len(subset_data) / total_samples
        avg_entropy += weight * calculate_entropy(subset_data)

    return avg_entropy


def calculate_information_gain(data, attribute_name):
    """
    Calculate information gain for a specific attribute.

    Args:
    - data: DataFrame, input data
    - attribute_name: str, attribute column name

    Returns:
    - info_gain: float, information gain value
    """
    # Calculate information gain using entropy and average entropy
    return calculate_entropy(data) - calculate_average_entropy(data, attribute_name)


def build_decision_tree(data, attribute_names):
    """
    Build a decision tree using the entropy method.

    Args:
    - data: DataFrame, input data
    - attribute_names: list, list of attribute column names

    Returns:
    - tree: dict, decision tree representation
    """
    if len(data.iloc[:, -1].unique()) == 1:
        return data.iloc[0, -1]  # If all samples have the same label, return the label

    if len(attribute_names) == 0:
        majority_label = data.iloc[:, -1].mode()[0]
        return majority_label  # If no more attributes, return majority class

    best_attribute = "NONE"
    max_info_gain = -1 

    # Choose attribute with maximum information gain
    for attribute_name in attribute_names:
        info_gain = calculate_information_gain(data, attribute_name)
        if info_gain > max_info_gain:
            max_info_gain = info_gain 
            best_attribute = attribute_name
    
    assert(best_attribute != "NONE")

    tree = {best_attribute: {}}
    remaining_attributes = attribute_names.copy()
    remaining_attributes.remove(best_attribute)

    # Recursively build tree based on best attribute
    for value in data[best_attribute].unique():
        subset_data = data[data[best_attribute] == value]
        tree[best_attribute][value] = build_decision_tree(subset_data, remaining_attributes)

    return tree


node_id = 0
def add_nodes_edges_to_tree(tree_graph, dot_graph=None):
    """
    Add nodes and edges to the Graphviz Digraph.

    Args:
    - tree_graph: dict, decision tree representation
    - dot_graph: Digraph, Graphviz Digraph object

    Returns:
    - dot_graph: Digraph, updated Graphviz Digraph object
    """
    if dot_graph is None:
        dot_graph = Digraph()

    global node_id
    for key, value in tree_graph.items():
        if isinstance(value, dict):
            for k in value.keys():
                dot_graph.node(k, k)  # Add node for each key
                dot_graph.edge(key, k)  # Add edge from key to node
            add_nodes_edges_to_tree(value, dot_graph)  # Recursively add nodes and edges
        else:
            dot_graph.node(str(node_id), value)  # Add leaf node with value
            dot_graph.edge(key, str(node_id))  # Add edge from key to leaf node
            node_id += 1  # Increment node ID

    return dot_graph


def visualize_decision_tree(tree_graph, filename='decision_tree'):
    """
    Visualize the decision tree and save as a PNG file.

    Args:
    - tree_graph: dict, decision tree representation
    - filename: str, output filename without extension
    """
    dot_graph = Digraph()
    dot_graph = add_nodes_edges_to_tree(tree_graph, dot_graph)
    dot_graph.render(filename, format='png', cleanup=True)  # Render and save decision tree as PNG


# Load data and describe statistics
data = load_data_and_describe("/Users/johnayman/Library/Mobile Documents/com~apple~CloudDocs/Desktop/EDU/FCAI-3/SECOND TERM/Supervised learning/Assignments/ASSIGNMENT 3/play_tennis.csv")
attribute_names = data.columns[:-1].tolist()  # Get attribute column names

# Build the decision tree
decision_tree = build_decision_tree(data, attribute_names)
print("OUTPUT TREE SHAPE: ")
print(decision_tree)
print("--------------------------------------")

# Visualize and save the decision tree
visualize_decision_tree(decision_tree)
print("Visualization Generated as PNG file")

