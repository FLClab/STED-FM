import numpy
import numpy as np 
from typing import List, Dict
import pickle

import tifffile
from collections.abc import Hashable

import sys
sys.path.insert(0, "../../")
from DEFAULTS import COLORS

def ask_user(current, other):
    import matplotlib.pyplot as plt

    choice = None

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(current.item, cmap='gray')
    axes[0].set_title('1')
    axes[1].imshow(other.item, cmap='gray')
    axes[1].set_title('2')

    ax_button1 = plt.axes([0.25, 0.05, 0.1, 0.075])
    ax_button2 = plt.axes([0.65, 0.05, 0.1, 0.075])
    button1 = Button(ax_button1, '1')
    button2 = Button(ax_button2, '2')

    def choose1(event):
        choice = '1'
        plt.close()

    def choose2(event):
        choice = '2'
        plt.close()

    button1.on_clicked(choose1)
    button2.on_clicked(choose2)

    plt.show()

# class Annotation:
#     def __init__(self, filename: str, image: np.ndarray, predictions: Dict[np.ndarray]):
#         self.choices = ["ImageNet", "HPA", "JUMP", "STED"]
#         assert list(predictions.keys()) == self.choices 
#         self.predictions = predictions
#         self.annotation = {}

#     def check_complete(self):
#         return len(set(self.annotation.keys())) == 4
    
#     def sample_pair(self):
#         if len(self.annotation.keys()) == 0:
#             keys = np.random.choice(self.choices, 2)

#         else:
#             k1 = np.random.choice(list(self.annotation.keys()))
#             choices = list(set(self.choices) - set(list(self.annotation.keys())))
#             k2 = np.random.choice(choices)
#             keys = [k1, k2]

#         img1, img2 = [self.predictions[key] for key in keys]
#         return (img1, img2), keys 

#     def add(self, keys: List[str], choice: str):
#         if len(self.annotations.keys()) == 0:
#             self.annotation[0] = choice
#             keys.remove(choice)
#             assert len(keys) == 1
#             self.annotation[1] = keys[0]
#         else:
#             pass
#             #TODO

# if __name__ == "__main__":


# import random
# from matplotlib.widgets import Button

class Queue(dict):
    def __init__(self) -> None:
        self.queue = []
    
    def enqueue(self, item):
        self.queue.append(item)
    
    def dequeue(self):
        return self.queue.pop(0)
    
    def clear(self):
        self.queue.clear()

    def __len__(self):
        return len(self.queue)

# queue = Queue()

class Tree:
    """
    A class used to represent a Tree structure.
    Attributes
    ----------
    root : Node, optional
        The root node of the tree (default is None).
    nodes : dict
        A dictionary to store nodes with their IDs as keys.
    Methods
    -------
    add_node(node)
        Adds a single node to the tree.
    add_nodes(nodes)
        Adds multiple nodes to the tree.
    contains(node)
        Checks if a node is already in the tree.
    traverse()
        Traverses the tree starting from the root node.
    get_ranking()
        Returns a list of nodes in the tree based on traversal.
    save(path)
        Saves the tree nodes to a file.
    load(path)
        Loads the tree nodes from a file.
    __len__()
        Returns the number of nodes in the tree.
    __repr__()
        Returns a string representation of the tree.
    """
    def __init__(self, root=None, queue=None):
        self.root = root
        self.nodes = {}

        self.queue = queue

    def add_node(self, node):
        """
        Adds a node to the tree.
        This method adds a node to the tree if it does not already exist. If the tree is empty, the node becomes the root. 
        Otherwise, the node is compared with the root to determine its position in the tree.
        Args:
            node (Node): The node to be added to the tree.
        Returns:
            None
        Raises:
            None
        """

        if self.contains(node):
            print(f"Node {node} already exists in the tree.")
            return
        self.nodes[node.id] = node

        if self.root is None:
            self.root = node
            return
        
        self.root.compare(node, queue=self.queue)
    
    def add_nodes(self, nodes):
        """
        Adds multiple nodes to the current structure.
        Args:
            nodes (iterable): An iterable of nodes to be added.
        """

        for node in nodes:
            self.add_node(node)
    
    def contains(self, node):
        """
        Check if a node is present in the nodes collection.
        Args:
            node (Node): The node to check for presence in the collection.
        Returns:
            bool: True if the node is present, False otherwise.
        """

        return node.id in self.nodes

    def traverse(self):
        """
        Traverse through all nodes starting from the root node.
        Yields:
            node: The current node in the traversal.
        """
        
        for node in self.root.traverse():
            yield node

    def get_ranking(self):
        """
        Retrieves a ranking of nodes by traversing the structure.
        Returns:
            list: A list of nodes obtained from traversing the structure.
        """

        return [node for node in self.traverse()]

    def save(self, path):
        """
        Saves the current state of nodes to a specified file path using pickle.
        Args:
            path (str): The file path where the nodes will be saved.
        """
        
        pickle.dump(self.nodes, open(path, "wb"))

    def load(self, path):
        """
        Load the nodes from a pickle file and set the root node.
        Args:
            path (str): The file path to the pickle file containing the nodes.
        Raises:
            Exception: If there is an error loading the pickle file.
        """
        
        self.nodes = pickle.load(open(path, "rb"))
        self.root = list(self.nodes.values())[0]
    
        # Make sure that all the nodes are connected
        connected_nodes = [node.id for node in self.traverse()]
        remove_nodes = []
        for node in self.nodes.values():
            if node.id not in connected_nodes:
                remove_nodes.append(node.id)
        for node in remove_nodes:
            del self.nodes[node]
    
    def __len__(self):
        return len(self.nodes)
    
    def __repr__(self):
        
        return f"Tree({self.root})"

class Node:
    """
    A class representing a node in a binary search tree.
    Attributes:
    -----------
    item : Item
        The item stored in the node. If the item is not an instance of Item, it will be converted to an appropriate type.
    attrs : dict
        A dictionary of attributes associated with the node. If not provided, a default dictionary with an "id" key is created.
    greater_than : Node or None
        The right child node, representing items greater than the current node's item.
    smaller_than : Node or None
        The left child node, representing items smaller than the current node's item.
    Methods:
    --------
    id:
        Returns the unique identifier of the node.
    traverse():
        Yields nodes in an in-order traversal of the binary search tree.
    compare(other):
        Compares the current node's item with another node's item and places the other node in the correct position in the tree.
    __repr__():
        Returns a string representation of the node.
    """

    def __init__(self, item, attrs=None, callback=None):
        
        if not isinstance(item, Item):
            if isinstance(item, numpy.ndarray):
                self.item = ImageItem(item, callback=callback)
            else:
                self.item = Item(item, callback=callback)
        else:
            self.item = item

        if attrs is None:
            attrs = {
                "id": id(self)
            }
        if "id" not in attrs:
            attrs["id"] = id(self)

        self.attrs = attrs

        self.greater_than = None
        self.smaller_than = None

        if not isinstance(self.id, Hashable):
            raise ValueError("The 'id' attribute of the node must be hashable.")

    def get_item(self):
        if isinstance(self.item, Item):
            return self.item.item
        return self.item

    @property
    def id(self):
        """
        Retrieve the 'id' attribute from the object's attributes.
        Returns:
            str: The 'id' attribute of the object.
        """

        return self.attrs["id"]

    def traverse(self):
        """
        Traverse the binary tree in an in-order manner.
        Yields:
            The nodes of the binary tree in ascending order.
        """

        if self.smaller_than is not None:
            yield from self.smaller_than.traverse()
        yield self
        if self.greater_than is not None:
            yield from self.greater_than.traverse()
        
    def compare(self, other, queue=None):
        """
        Compare the current node with another node and insert the other node
        into the appropriate position in the binary tree.
        If the item of the current node is greater than or equal to the item
        of the other node, the other node is inserted into the 'greater_than'
        subtree. Otherwise, it is inserted into the 'smaller_than' subtree.
        Args:
            other (Node): The node to be compared and inserted.
        Returns:
            None
        """
        # This is the iterative version of the compare method
        if queue is not None and self.item.callback is None and not isinstance(self.get_item(), (int, float)):
            queue.enqueue((self, other))
            
        # This assumes that the user provided a callback function when creating the node
        # Since we pass the item when creating the node, we can ignore the callback function
        else:
            if self.item >= other.item:
                if self.greater_than is None:
                    self.greater_than = Node(other.item, attrs=other.attrs)
                else:
                    self.greater_than.compare(other)
            else:
                if self.smaller_than is None:
                    self.smaller_than = Node(other.item, attrs=other.attrs)
                else:
                    self.smaller_than.compare(other)

    def add_child(self, other, is_greater, queue=None):
        if is_greater:
            if self.greater_than is None:
                self.greater_than = Node(other.item, attrs=other.attrs)
            else:
                self.greater_than.compare(other, queue=queue)
        else:
            if self.smaller_than is None:   
                self.smaller_than = Node(other.item, attrs=other.attrs)
            else:
                self.smaller_than.compare(other, queue=queue)
    
    def __repr__(self):
        # string = f"Node({self.item})"
        # if self.smaller_than is not None:
        #     string = f"{self.smaller_than} <- {string}"
        # if self.greater_than is not None:
        #     string = f"{string} -> {self.greater_than}"
        # return string
        return f"Node({self.get_item()})"
    
class Item:
    """
    A class used to represent an Item.
    Attributes
    ----------
    item : any
        The value of the item.
    Methods
    -------
    __init__(item)
        Initializes the Item with the given value.
    __ge__(other)
        Compares this Item with another Item. If both items are numeric (int or float), 
        it returns the result of the comparison. Otherwise, it prompts the user to 
        select the greater item.
    __repr__()
        Returns a string representation of the Item.
    """

    def __init__(self, item, callback=None):
        self.item = item
        self.callback = callback

    def __ge__(self, other):
        if isinstance(self.item, (int, float)) and isinstance(other.item, (int, float)):
            return self.item >= other.item

        choice = self.callback(self.item, other.item)

        # question = f"Select the greater item: [1] {self.item} OR [2] {other.item}"
        # answer = input(question)
        # while answer not in ["1", "2"]:
        #     answer = input(question)
        return choice == "1"

    def __repr__(self):
        return f"Item({self.item})"

class ImageItem(Item):
    """
    ImageItem class that extends the Item class.
    Attributes:
        item: The item to be wrapped by the ImageItem class.
    Methods:
        __init__(item):
            Initializes the ImageItem with the given item.
        __ge__(other):
            Compares this ImageItem with another ImageItem using a user-defined choice.
        __repr__():
            Returns a string representation of the ImageItem, showing the shape of the item.
    """

    def __init__(self, item, callback=None):
        super().__init__(item, callback=callback)

    def __ge__(self, other):
        
        choice = self.callback(self.item, other.item)

        return choice == '1'

    def __repr__(self):
        return f"ImageItem({self.item.shape})"

def build_tree(items, tree=None, callback=None, queue=None):
    """
    Builds a tree structure from a list of items.
    Args:
        items (list): A list of items where each item can be a dictionary or any other type.
                        If an item is a dictionary, it should contain an "item" key, an "id" key and other attributes.
        tree (Tree, optional): An existing tree to which nodes will be added. If None, a new Tree is created.
    Returns:
        Tree: The constructed tree with nodes added from the items list.
    Example:
        items = [{"item": "root", "id": 0, "attr1": "value1"}, {"item": "child", "id": 1, "attr2": "value2"}]
        tree = build_tree(items)
    """

    if tree is None:
        tree = Tree(queue=queue)

    for item in items:
        if isinstance(item, dict):
            item, attrs = item["item"], item
        else:
            attrs = {}
        attrs["idx"] = len(tree)
        tree.add_node(Node(item, attrs=attrs, callback=callback))
    return tree

if __name__ == "__main__":

    import random
    import argparse
    import sys 
    import os
    import glob
    from matplotlib import pyplot
    from collections import defaultdict
    from itertools import combinations
    from scipy.stats import kendalltau
    sys.path.insert(0, "../../")
    from DEFAULTS import COLORS


    ranking_dict = glob.glob("/home/frederic/flc-dataset/experiments/segmentation-experiments/patch-retrieval-experiment/candidates/*", recursive=True)
    ranking_dict = [f.split("/")[-1].split(".")[0] for f in ranking_dict]
    ranking_dict = {f: i for i, f in enumerate(ranking_dict)}

    # numpy.random.seed(42)
    # random.seed(42)
    pretraining_datasets = ["STED", "SIM", "HPA","JUMP", "ImageNet"][::-1]

    scores = defaultdict(list)

    files = glob.glob("./data/patch-retrieval-experiment/*-tree.pkl")
    files = [f for f in files if "Anthony" not in f]
    files = [f for f in files if "Fred" not in f]
    USERS = []
    correlation_array = numpy.zeros((len(files), 10))
    for idx, f in enumerate(files):
        user = f.split("/")[-1].split(".")[0].split("-")[0]
        tree = Tree()
        tree.load(f)

        image_steps = []
        rankings = [item.get_item() for item in tree.get_ranking()]
        correlation = [item.split("/")[-1].split(".")[0] for item in rankings]
        correlation = [ranking_dict[item] for item in correlation]
        correlation_array[idx] = correlation[:10]
        print(f"=== {user} ===")
        USERS.append(user)
        for r in rankings[:10]:
            print(f"\t{r}")
        print("\n\n")
        rankings = [item.split("/")[-1].split(".")[0] for item in rankings]
        for pretraining in pretraining_datasets:
            curr_rankings = np.array([(pretraining.lower() in item.lower())*1 for item in rankings])[:10]
            target_label = 1
            average_precision = np.sum(curr_rankings == target_label) / len(curr_rankings)
            scores[pretraining].append(average_precision)
            
        for node in tree.get_ranking():
            # print(node, node.attrs)
            step = node.get_item()
            step = os.path.basename(step).split(".")[0].split("-")[-1]
            image_steps.append(step)

        image_steps = numpy.array(image_steps)

        fig, ax = pyplot.subplots(figsize=(4, 3))

        uniques = numpy.unique(image_steps)
        for unique in uniques:
            mask = image_steps == unique
            ax.plot(numpy.cumsum(mask) / mask.sum(), label=f"{unique}", color=COLORS[unique])
        ax.set(
            xlabel="Ranking",
            ylabel="Fraction of images"
        )
        ax.legend()
        fig.savefig(f"./ranking-{user}-patch-retrieval-experiment.pdf", dpi=300, bbox_inches="tight")

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    data = list(scores.values())
    keys = list(scores.keys())
    boxes = ax.boxplot(data, labels=keys, patch_artist=True)
    
    # Style each box
    for key, box in zip(keys, boxes['boxes']):
        box.set_facecolor(COLORS[key])  
        box.set_color(COLORS[key])     
        box.set_alpha(0.6) 
        box.set_edgecolor("black")           
    
    # Style other elements
    pyplot.setp(boxes['medians'], color='black')
    pyplot.setp(boxes['whiskers'], color='black') 
    pyplot.setp(boxes['caps'], color='black')   
    pyplot.setp(boxes['fliers'], markerfacecolor='black', marker='o') 
    ax.set_ylabel("% in top 10")
    fig.savefig("./patch-retrieval-boxplot.pdf", dpi=1200, bbox_inches="tight")
    pyplot.close(fig)

    ## Correlation code 
    # correlation_array = numpy.array(correlation_array)
    # print(correlation_array.shape)
    # user_pairs = list(combinations(range(correlation_array.shape[0]), 2))
    # correlation_matrix = numpy.ones((len(USERS), len(USERS)))
    # for (u1, u2) in user_pairs:
    #     arr1 = correlation_array[u1]
    #     arr2 = correlation_array[u2]
    #     print(arr1.shape, arr2.shape)
    #     res = kendalltau(arr1, arr2) 
    #     tau = res.statistic
    #     print(f"{USERS[u1]} and {USERS[u2]} have a Kendall Tau of {tau}")
    #     correlation_matrix[u1, u2] = tau
    #     correlation_matrix[u2, u1] = tau

    # fig = pyplot.figure()
    # ax = fig.add_subplot(111)
    # im = ax.imshow(correlation_matrix, cmap="RdPu")
    
    # # Add correlation values to each cell
    # for i in range(len(USERS)):
    #     for j in range(len(USERS)):
    #         color = "white" if correlation_matrix[i, j] > 0.5 else "black"
    #         text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
    #                      ha="center", va="center", color="black")
    
    # ax.set_xticks(range(len(USERS)))
    # ax.set_yticks(range(len(USERS)))
    # ax.set_xticklabels(USERS, rotation=90)
    # ax.set_yticklabels(USERS)
    # fig.savefig("correlation_matrix.pdf", dpi=1200, bbox_inches="tight")
    # pyplot.close(fig)
    

    # items = [numpy.random.rand(128, 128) for _ in range(5)]
    # # items = [2,1,5,3,9,8]
    # items = [{"item": item, "id": i} for i, item in enumerate(items)]

    # tree = build_tree(items)
    # print(len(tree.get_ranking()))



    # tree.save("tree.pkl")
    # print("Tree saved.")

    # print(queue.queue)

    # del tree

    # Multidomain ranking

    # tree = Tree()
    # tree.load("data/multidomain/Anthony-tree.pkl")
    # print(len(tree))
    # import os
    # from matplotlib import pyplot

    # image_steps = []
    # for node in tree.get_ranking():
    #     # print(node, node.attrs)
    #     step = node.get_item()
    #     step = os.path.basename(step).split(".")[0].split("_")[-1]
    #     image_steps.append(int(step))

    # fig, ax = pyplot.subplots(figsize=(4, 3))
    # image_steps = numpy.array(image_steps)
    # cmap = pyplot.get_cmap("RdPu", 1 + len(numpy.unique(image_steps)))
    # for idx in numpy.unique(image_steps):
    #     mask = image_steps == idx
    #     ax.plot(numpy.cumsum(mask) / mask.sum(), label=f"Step {idx}", color=cmap(idx + 1))
    # ax.set(
    #     xlabel="Ranking",
    #     ylabel="Fraction of images"
    # )
    # ax.legend()
    # fig.savefig("ranking.pdf", dpi=300, bbox_inches="tight")

    # print(image_steps)

    # Patch retrieval ranking
    tree = Tree()
    tree.load("data/patch-retrieval-experiment/jujubee-tree.pkl")
    print(len(tree))

    import os
    from matplotlib import pyplot

    image_steps = []
    for node in tree.get_ranking():
        # print(node, node.attrs)
        step = node.get_item()
        step = os.path.basename(step).split(".")[0].split("-")[-1]
        image_steps.append(step)

    image_steps = numpy.array(image_steps)

    fig, ax = pyplot.subplots(figsize=(4, 3))

    uniques = numpy.unique(image_steps)
    for unique in uniques:
        mask = image_steps == unique
        ax.plot(numpy.cumsum(mask) / mask.sum(), label=f"{unique}", color=COLORS[unique])
    ax.set(
        xlabel="Ranking",
        ylabel="Fraction of images"
    )
    ax.legend()
    fig.savefig("ranking-patch-retrieval-experiment.pdf", dpi=300, bbox_inches="tight")

    # items = [2,5, 1, 3,9,8, 6, 4, 7]
    # items = [{"item": i, "id": i} for i in items]

    # tree = build_tree(items, tree)

    # ranking = tree.get_ranking()
    # for node in ranking:
    #     print(node, node.attrs)


    # import tifffile
    # import tiffwrapper
    # import os
    # from PIL import Image
    # import shutil

    # weights = ["MAE_SMALL_IMAGENET1K_V1", "MAE_SMALL_STED"]
    # class_id = "perforated"

    # shutil.rmtree(os.path.join(".", "application", "static", class_id))

    # os.makedirs(os.path.join(".", "application", "static", class_id), exist_ok=True)
    # os.makedirs(os.path.join(".", "application", "static", class_id, "candidates"), exist_ok=True)
    # for weight in weights:

    #     template = tifffile.imread(f"/home/anthony/Documents/flc-dataset/experiments/detection-experiments/template-{class_id}-{weight}.tif")
    #     composite = tiffwrapper.make_composite(template[[0]], ["hot"])
        
    #     image = Image.fromarray(composite.astype('uint8'))
    #     image.save(f"application/static/{class_id}/template.png")

    #     images = tifffile.imread(f"/home/anthony/Documents/flc-dataset/experiments/detection-experiments/crop-{class_id}-0-{weight}.tif")
    #     for i, image in enumerate(images[:10]):
    #         composite = tiffwrapper.make_composite(image[numpy.newaxis, ...], ["hot"])
    #         image = Image.fromarray(composite.astype('uint8'))
            
    #         image.save(f"application/static/{class_id}/candidates/{weight}-{i}.png")
    
