
from annotation import Tree

tree = Tree()
tree.load("application/Fred -perforated-tree.pkl")
# tree.load("application/guest-perforated-tree.pkl")

ranking = tree.get_ranking()

for node in ranking:
    print(node.get_item())