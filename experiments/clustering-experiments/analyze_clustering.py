import numpy as np 
import pickle 
import os
import matplotlib.pyplot as plt 
from typing import List, Tuple, Dict, Any, Optional
import argparse 
from matplotlib import colormaps
import networkx 
from scipy.spatial import distance 
import pydot 
from matplotlib.patches import Patch 
from networkx.drawing.nx_agraph import graphviz_layout

parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str, default=None)
parser.add_argument("--dataset", type=str, default="neural-activity-states")
parser.add_argument("--depth", type=int, default=20)
parser.add_argument("--mode", type=str, default="deep", choices=["deep", "manual", "morphological"])
args = parser.parse_args()

def get_colormap(condition: str):
    if condition == "Block":
        return "Greys"
    elif condition == "48hTTX":
        return "Reds"
    elif condition == "0MgGlyBic":
        return "Blues"
    elif condition == "GluGly":
        return "Greens"

class Node:
    def __init__(self, cluster_id, depth, data=None):
        self.cluster_id = cluster_id
        self.depth = depth
        self.data = data
        self.parent = None
        self.children = []
    
    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)
    
    def __str__(self):
        return f"Node(cluster_id={self.cluster_id}, depth={self.depth}, children={len(self.children)})"
    
    def __repr__(self):
        return self.__str__()

def print_tree(node, indent=0):
    """Helper function to visualize the tree structure"""
    print("  " * indent + str(node))
    print(f"DATA: {node.data}\n\n")
    if node.data is not None:
        print("  " * (indent + 1) + f"Data: {node.data}")
    for child in node.children:
        print_tree(child, indent + 1)
        
def build_tree_from_nested_lists(nested_lists, depth=0, parent_id=0):
    """
    Recursively builds a tree from nested lists.
    
    Args:
        nested_lists: A list that may contain nested lists or data elements
        depth: Current depth in the tree
        parent_id: Cluster ID of the parent node
    
    Returns:
        The root node of the tree
    """
    # Create a root node for the current level
    current_id = parent_id
    root = Node(cluster_id=current_id, depth=depth)
    
    # Process each item in the nested list
    for i, item in enumerate(nested_lists):
        # Assign a unique cluster ID for the current node
        current_id = parent_id * 100 + i + 1
        
        if isinstance(item, list):
            # Recursive case: item is a nested list (cluster)
            child_node = build_tree_from_nested_lists(item, depth + 1, current_id)
            root.add_child(child_node)
            # print(f"child_node: {child_node.parent}")
        else:
            # Base case: item is data
            leaf_node = Node(cluster_id=current_id, depth=depth + 1, data=item)
            root.add_child(leaf_node)
            # print(f"leaf_node: {leaf_node.parent}")
    
    return root

def load_data(data_path: str):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

def find_leaf_nodes(node, leaves=None):
    if leaves is None:
        leaves = []
    
    if not node.children:  # This is a leaf node
        leaves.append(node)
    else:
        for child in node.children:
            find_leaf_nodes(child, leaves)
            
    return leaves

def get_depth_nodes(root, depth):
    for child in root.children:
        if child.depth == depth:
            yield child
        else:
            yield from get_depth_nodes(child, depth)

def extract_mean_feature_vector(node):
    leaf_nodes = find_leaf_nodes(node)
    all_data = np.concatenate([leaf.data["data"] for leaf in leaf_nodes])
    return np.mean(all_data, axis=0)

def plot_frequencies(node_list, display: bool = False):
    num_clusters = len(node_list)
    states = ["Block", "48hTTX", "0MgGlyBic", "GluGly"] if args.dataset == "neural-activity-states" else ["4hMeOH", "6hMeOH", "8hMeOH", "16hMeOH"]

    cluster_counts = {condition: {key: 0 for key in range(num_clusters)} for condition in states}
    condition_counts = {key: {condition: 0 for condition in states} for key in range(num_clusters)}


    for node_idx, node in enumerate(node_list):
        leaf_nodes = find_leaf_nodes(node)
        for leaf in leaf_nodes:
            all_metadata = leaf.data["metadata"]
            if args.dataset == "neural-activity-states":
                labels = [item["condition"] for item in all_metadata]
            else:
                labels = [item["label"] for item in all_metadata]
                labels = [states[item.item()] for item in labels]
            uniques, counts = np.unique(labels, return_counts=True)
            for u, c in zip(uniques, counts):
                cluster_counts[u][node_idx] += c
                condition_counts[node_idx][u] += c

    if display:
        r = np.arange(1, num_clusters+1)
        ogr = np.arange(1, num_clusters+1)
        width = 0.2
        H = 10 if num_clusters < 10 else 20
        fig = plt.figure(figsize=(num_clusters * 3, H))
        ax = fig.add_subplot(111)
        cmap = colormaps.get_cmap("RdPu")
        colors = cmap(np.linspace(0.3, 1.0, len(cluster_counts)))
        for idx, condition in enumerate(states):
            height = [cluster_counts[condition][i] for i in range(num_clusters)]
            ax.bar(r, height, width=width, label=condition, align="edge", color=colors[idx])
            r = [x + width for x in r]
        ax.set(
            title="Cluster counts",
            xticks=[item + (width * 4/2) for item in ogr],
            xticklabels=list(range(num_clusters)),
        )
        ax.tick_params(axis='both', labelsize=10 + num_clusters)  # Set fontsize of the tick labels
        ax.legend()
        plt.show()
    return condition_counts

def compute_similarity(vector1, vector2):
    similarity = 1 - distance.cosine(vector1, vector2)  
    return similarity

def get_color(counts: dict, root: bool = False):
    if args.condition is None and root:
        return "black", True
    elif args.condition is None and not root:
        if args.dataset == "neural-activity-states":
            color_dict = {"Block": 0, "48hTTX": 1, "0MgGlyBic": 2, "GluGly": 3}
        else:
            color_dict = {"4hMeOH": 0, "6hMeOH": 1, "8hMeOH": 2, "16hMeOH": 3}
        keys = list(counts.keys())
        values = list(counts.values())
        max_key = keys[np.argmax(values)]
        if args.dataset == "neural-activity-states":
            colors = {0: "grey", 1: "hotpink", 2: "dodgerblue", 3: "goldenrod"}
        else:
            colors = {0: "grey", 1: "hotpink", 2: "dodgerblue", 3: "goldenrod"}
        return colors[color_dict[max_key]], True
    else:
        class_count = counts[args.condition]
        total_count = sum(counts.values())
        class_proportion = class_count / total_count
        cmap = colormaps.get_cmap("RdPu")
        color_value = 0.3 + class_proportion
        if class_proportion > 0.0:
            return cmap(color_value), True 
        else:
            return cmap(color_value), False

def display_graph(graph, node_size_scale_factor=10, edge_weight_factor=5.0, min_edge_width=1.0):
    # Get node colors
    cmap = colormaps.get_cmap("RdPu")
    colors_per_condition = cmap(np.linspace(0.0, 1.0, 4)) if args.condition is not None else ["grey", "hotpink", "dodgerblue", "goldenrod"]
    colors = [node[1]["color"] for node in graph.nodes(data=True)]
    
    edge_weights = np.array([graph[u][v]['weight'] for u, v in graph.edges()])
    # print(edge_weights)
    edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())
    # edge_weights = [item + 1 for item in edge_weights]

    edge_widths = [min_edge_width + (weight * edge_weight_factor) for weight in edge_weights]
    # edge_widths = [min_edge_width]
    node_sizes = np.array([node[1]["count"] * node_size_scale_factor for node in graph.nodes(data=True)])
    node_sizes = np.clip(node_sizes, 10, 800)
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    

    # layout = networkx.kamada_kawai_layout(graph)
    layout = graphviz_layout(graph, prog="twopi")
    if args.condition is not None:
        node_colors = [node[1]["color"] if node[1]["add"] else "gainsboro" for node in graph.nodes(data=True)]
        edge_colors = ["gray" if edge[2]["add"] else "gainsboro" for edge in graph.edges(data=True)]
        networkx.draw_networkx_nodes(graph, pos=layout, node_color=node_colors, node_size=node_sizes)
        networkx.draw_networkx_edges(graph, pos=layout, width=edge_widths, 
                                alpha=0.7, edge_color=edge_colors)
    else:
        networkx.draw_networkx_nodes(graph, pos=layout, node_color=colors, node_size=node_sizes)
        # networkx.draw_networkx_labels(graph, pos=layout)
        networkx.draw_networkx_edges(graph, pos=layout, width=edge_widths, 
                                alpha=0.7, edge_color='gray')
    
    
    # # Remove axis
    # if args.condition is None:
    #     if args.dataset == "neural-activity-states":
    #         conditions = ["Block", "48hTTX", "0MgGlyBic", "GluGly"]
    #     else:
    #         conditions = ["4hMeOH", "6hMeOH", "8hMeOH", "16hMeOH"]
    #     legend_elements = [
    #         Patch(facecolor=colors_per_condition[i], label=conditions[i]) for i in range(4)
    #     ]
    #     ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))
    # # ax.axis("off")
    plt.tight_layout()
    plt.show()
    os.makedirs(f"./graphs-{args.mode}", exist_ok=True)
    fig.savefig(f"./graphs-{args.mode}/{args.condition}_{args.dataset}_graph.pdf", dpi=1200, bbox_inches="tight")
    # os.makedirs("./trees", exist_ok=True)
    # fig.savefig(f"./trees/{args.mode}_{args.condition}_{args.dataset}_graph.pdf", dpi=1200, bbox_inches="tight")
    

if __name__ == "__main__":
    if args.mode == "deep":
        data_path = "./recursive-clustering-experiment/deep/MAE_SMALL_STED_neural-activity-states_recursive_clusters_tree.pkl" 
    elif args.mode == "manual":
        data_path = "./recursive-clustering-experiment/manual/MAE_SMALL_STED_neural-activity-states_recursive_clusters_tree.pkl"
    elif args.mode == "morphological":
        data_path = "./recursive-clustering-experiment/morphological/MAE_SMALL_STED_neural-activity-states_recursive_clusters_tree.pkl"
    data = load_data(data_path)
    
    tree = build_tree_from_nested_lists(data)

  
    all_nodes = find_leaf_nodes(tree)

    max_depth = max([node.depth for node in all_nodes])
    # print_tree(tree)
    mean_vector = extract_mean_feature_vector(tree)
    graph = networkx.DiGraph()
    all_counts = plot_frequencies([tree], display=False)
    color, _ = get_color(all_counts[0], root=True)
    graph.add_node(tree.cluster_id, color=color, count=sum(list(all_counts[0].values())), add=True)

    for d in range(1, min(args.depth + 1, max_depth + 1)):
        print(f"=== Depth {d} ===")
        depth_nodes = list(get_depth_nodes(tree, d))
        depth_counts = plot_frequencies(depth_nodes, display=False)

        for k, node in zip(depth_counts.keys(), depth_nodes):
            node_vector = extract_mean_feature_vector(node)
            parent_vector = extract_mean_feature_vector(node.parent)
            similarity = compute_similarity(node_vector, parent_vector)
            if args.condition is None:
                node_count = max(list(depth_counts[k].values()))
            else:
                node_count = sum(list(depth_counts[k].values()))
            color, add_node = get_color(depth_counts[k], root=False)
            
            graph.add_node(node.cluster_id, color=color, count=node_count, add=add_node)
            graph.add_edge(node.parent.cluster_id, node.cluster_id, weight=similarity, add=add_node)

    display_graph(graph)



    
    