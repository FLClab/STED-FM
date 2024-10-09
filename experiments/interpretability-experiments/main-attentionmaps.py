import timm 
from timm.models.layers import PatchEmbed 
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import argparse
import sys 
sys.path.insert(0, "../")
from DEFAULT import BASE_PATH
from loaders import get_dataset 
from model_builder import get_pretrained_model_v2
from utils import SaveBestModel, AverageMeter, update_cfg, get_number_of_classes 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="optim")
parser.add_arguemnt("--model", type=str, default="mae-small_STED")
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--opts", nargs="+", default=[])
args = parser.parse_args()

def main():
    pass 

if __name__=="__main__":
    main()