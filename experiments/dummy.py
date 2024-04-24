import torch
import numpy as np
from model_builder import get_pretrained_model
import argparse 
import torchvision

def list_of_numbers(arg):
    return [int(item) for item in arg.split(',')]

parser = argparse.ArgumentParser()
parser.add_argument("--freeze-blocks", type=list_of_numbers)
args = parser.parse_args()

def main():
    # print(type(args.freeze_blocks))
    # print(f"{args.freeze_blocks}\n\n")
    # model = get_pretrained_model(
    #     name="MAEClassifier",
    #     weights="STED",
    #     global_pool='avg',
    #     blocks=args.freeze_blocks
    # )
    resnet = torchvision.models.resnet18(weights=None)
    for n, p in resnet.named_parameters():
        print(n, p)

if __name__=="__main__":
    main()