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
from skimage import measure
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial import distance
import sys
from tqdm.auto import tqdm

from sklearn.metrics import DistanceMetric 
from stedfm.loaders import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="./recursive-clustering-experiment/manual/MAE_SMALL_STED_neural-activity-states_recursive_clusters_tree.pkl")
parser.add_argument("--condition", type=str, default=None)
parser.add_argument("--dataset", type=str, default="neural-activity-states")
parser.add_argument("--depth", type=int, default=20)
args = parser.parse_args()

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

def compute_similarity(vector1, vector2):
    similarity = 1 - distance.cosine(vector1, vector2)  
    return similarity

def extract_mean_feature_vector(node):
    leaf_nodes = find_leaf_nodes(node)
    all_data = np.concatenate([leaf.data["data"] for leaf in leaf_nodes])
    return np.mean(all_data, axis=0)


def get_average_features(node, dataset):
    leaf_nodes = find_leaf_nodes(node)
    total = 0
    area, intensity, eccentricity, nn, num_proteins, density, blur_effect, shannon_entropy, signal_to_noise = [], [], [], [], [], [], [], [], []
    for leaf in leaf_nodes:
        total += leaf.data["data"].shape[0]
        try:
            data_idx = [item["dataset-idx"].item() for item in leaf.data["metadata"]]
        except:
            data_idx = [item["dataset-idx"] for item in leaf.data["metadata"]]
        imgs = [dataset[idx][0].squeeze().numpy() for idx in data_idx]
        masks = [dataset[idx][1]["mask"] for idx in data_idx]
        for img, mask in zip(imgs, masks):
            label_image, nprots = measure.label(mask, return_num=True)
            # Image-level features
            num_proteins.append(nprots)
            br = measure.blur_effect(img)
            blur_effect.append(br)
            shannon_entropy.append(measure.shannon_entropy(img))
            foreground_intensity = np.mean(img[mask])
            inverted_mask = np.logical_not(mask)
            background_intensity = np.mean(img[inverted_mask])
            signal_to_noise.append(foreground_intensity / background_intensity)
            # Protein-level features
            # img_density = []
            # for y in np.arange(0, img.shape[0], 16):
            #     for x in np.arange(0, img.shape[1], 16):
            #         crop_proteins = np.unique(label_image[y:y+16, x:x+16])
            #         img_density.append(len(crop_proteins) / (16 * 16))
            # density.append(np.mean(img_density))
            rprops = measure.regionprops(label_image, intensity_image=img)
            centroids = [r.weighted_centroid for r in rprops]
            if len(centroids) == 1:
                density.append(1.0)
            else:
                distance_matrix = distance.cdist(centroids, centroids, metric="euclidean")
                distance_matrix = np.sort(distance_matrix, axis=1)
                img_density = []
                for d in range(distance_matrix.shape[0]):
                    num_neighbors = np.sum(distance_matrix[d] < 50)
                    img_density.append(num_neighbors)
                density.append(np.mean(img_density))
                
            img_area, img_intensity, img_eccentricity, img_nn = [], [], [], []
            for rprop in rprops:
                distances = distance.cdist(centroids, [rprop.weighted_centroid], metric="euclidean")
                distances = distances[1:]
                if distances.shape[0] > 0:
                    img_nn.append(np.min(distances))
                img_area.append(rprop.area)
                img_intensity.append(rprop.mean_intensity)
                img_eccentricity.append(rprop.eccentricity)
            area.append(np.mean(img_area))
            intensity.append(np.mean(img_intensity))
            eccentricity.append(np.mean(img_eccentricity))
            nn.append(np.mean(img_nn))
    nn = [value for value in nn if not np.isnan(value)]
    area = np.mean(area)
    intensity = np.mean(intensity)
    eccentricity = np.mean(eccentricity)
    nn = np.mean(nn)
    num_proteins = np.mean(num_proteins)
    density = np.mean(density)
    blur_effect = np.mean(blur_effect)
    shannon_entropy = np.mean(shannon_entropy)
    signal_to_noise = np.mean(signal_to_noise)
    average_features = np.array([area, intensity, eccentricity, nn, num_proteins, density, blur_effect, shannon_entropy, signal_to_noise])
    
    return average_features, total

def display_graph(graph, feature: str, node_size_scale_factor=10, edge_weight_factor=5.0, min_edge_width=1.0):
    # Get node colors
    feature_values = [node[1][feature] for node in graph.nodes(data=True)]
    feature_values = np.array(feature_values)
    feature_values = (feature_values - np.quantile(feature_values, 0.05)) / (np.quantile(feature_values, 0.95) - np.quantile(feature_values, 0.05))
    feature_values = np.clip(feature_values, 0.3, 1.0)
    cmap = colormaps.get_cmap("RdPu")
    colors = cmap(feature_values)
    
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
   
    networkx.draw_networkx_nodes(graph, pos=layout, node_color=colors, node_size=node_sizes)
    networkx.draw_networkx_edges(graph, pos=layout, width=edge_widths, 
                                alpha=0.7, edge_color='gray')
    
    
    # Remove axis
    # ax.axis("off")
    plt.tight_layout()
    plt.show()
    os.makedirs("./graphs-manual", exist_ok=True)
    fig.savefig(f"./graphs-manual/test_{feature}_{args.dataset}_graph.pdf", dpi=1200, bbox_inches="tight")

if __name__=="__main__":
    data = load_data(args.data_path)
    tree = build_tree_from_nested_lists(data)
    all_nodes = find_leaf_nodes(tree)
    max_depth = max([node.depth for node in all_nodes])

    _, _, test_loader = get_dataset(
        name=args.dataset,
        transform=None,
        training=True,
        patch=None,
        batch_size=64,
        n_channels=1,
        balance=False,
        classes=["Block", "0MgGlyBic", "GluGly", "48hTTX"]
    )
    dataset = test_loader.dataset
    
    graph = networkx.DiGraph()
    root_features, root_count = get_average_features(tree, dataset)
    graph.add_node(
        tree.cluster_id, 
        area=root_features[0], 
        intensity=root_features[1], 
        eccentricity=root_features[2], 
        nn=root_features[3], 
        num_proteins=root_features[4],
        density=root_features[5],
        blur_effect=root_features[6],
        shannon_entropy=root_features[7],
        signal_to_noise=root_features[8],
        count=root_count
    )
    
    for d in range(1, min(args.depth + 1, max_depth + 1)):
        depth_nodes = list(get_depth_nodes(tree, d))
        for node in tqdm(depth_nodes, desc=f"Depth {d}"):
            node_features, node_count = get_average_features(node, dataset)
            node_vector = extract_mean_feature_vector(node)
            parent_vector = extract_mean_feature_vector(node.parent)
            similarity = compute_similarity(node_vector, parent_vector)
            graph.add_node(
                node.cluster_id, 
                area=node_features[0], 
                intensity=node_features[1], 
                eccentricity=node_features[2], 
                nn=node_features[3], 
                num_proteins=node_features[4],
                density=node_features[5],
                blur_effect=node_features[6],
                shannon_entropy=node_features[7],
                signal_to_noise=node_features[8],
                count=node_count,
            )
            graph.add_edge(
                node.parent.cluster_id,
                node.cluster_id,
                weight=similarity
            )

    for i, feature in enumerate(["area", "intensity", "eccentricity", "nn", "num_proteins", "density", "blur_effect", "shannon_entropy", "signal_to_noise"]):
        display_graph(graph, feature)