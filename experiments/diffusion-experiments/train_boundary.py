import numpy as np 
import matplotlib.pyplot as plt
import torch 
from sklearn import svm
from typing import Tuple
import argparse 
import pickle
import os

from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, "../")
from utils import set_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="quality")
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--channel", type=str, default="FUS")
args = parser.parse_args()

PATH = f"./{args.dataset}-experiment/embeddings"


def load_embedding(path: str ) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    embeddings, labels = data["embeddings"], data["labels"]
    return embeddings, labels


def main():

    set_seeds(args.seed)

    if args.dataset == "als":
        train_embeddings, train_labels = load_embedding(f"{PATH}/{args.weights}-{args.dataset}-embeddings_train_{args.channel}.npz")
        num_train_samples, latent_dim = train_embeddings.shape
        valid_embeddings, valid_labels = load_embedding(f"{PATH}/{args.weights}-{args.dataset}-embeddings_valid_{args.channel}.npz")
        num_valid_samples, _ = valid_embeddings.shape
    else:
        train_embeddings, train_labels = load_embedding(f"{PATH}/{args.weights}-{args.dataset}-embeddings_train.npz")
        num_train_samples, latent_dim = train_embeddings.shape
        valid_embeddings, valid_labels = load_embedding(f"{PATH}/{args.weights}-{args.dataset}-embeddings_valid.npz")
        num_valid_samples, _ = valid_embeddings.shape

    C = 1.0
    clf = svm.SVC(kernel="linear", C=C, class_weight="balanced")
    clf.fit(train_embeddings, train_labels) 

    val_prediction = clf.predict(valid_embeddings)
    print(np.unique(val_prediction, return_counts=True))
    accuracy = np.sum(val_prediction == valid_labels) / num_valid_samples
    print(f"Validation accuracy: {accuracy}")
    print(f"Confusion matrix: \n{confusion_matrix(valid_labels, val_prediction)}")

    boundary = clf.coef_.reshape(1, latent_dim).astype(np.float32)
    norm = np.linalg.norm(boundary)
    intercept = clf.intercept_ / norm
    boundary = boundary / norm

    os.makedirs(f"./{args.dataset}-experiment/boundaries", exist_ok=True)

    if args.dataset == "als":
        with open(f"./{args.dataset}-experiment/boundaries/{args.weights}_{args.dataset}_svm_{args.channel}.pkl", "wb") as f:
            pickle.dump(clf, f)
        np.savez(f"./{args.dataset}-experiment/boundaries/{args.weights}_{args.dataset}_boundary_{args.channel}.npz", boundary=boundary, intercept=intercept, norm=norm)
    else:
        with open(f"./{args.dataset}-experiment/boundaries/{args.weights}_{args.dataset}_svm.pkl", "wb") as f:
            pickle.dump(clf, f)
        np.savez(f"./{args.dataset}-experiment/boundaries/{args.weights}_{args.dataset}_boundary.npz", boundary=boundary, intercept=intercept, norm=norm)




if __name__ == "__main__":
    main()