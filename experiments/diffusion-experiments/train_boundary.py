import numpy as np 
import matplotlib.pyplot as plt
import torch 
from sklearn import svm
from typing import Tuple
import argparse 

PATH = "./lerp-results/embeddings/quality"

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, default="MAE_SMALL_STED")
args = parser.parse_args()


def load_embedding(path: str ) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    embeddings, labels = data["embeddings"], data["labels"]
    return embeddings, labels


def main():
    train_embeddings, train_labels = load_embedding(f"{PATH}/{args.weights}-optimquality-embeddings_train.npz")
    num_train_samples, latent_dim = train_embeddings.shape
    valid_embeddings, valid_labels = load_embedding(f"{PATH}/{args.weights}-optimquality-embeddings_valid.npz")
    num_valid_samples, _ = valid_embeddings.shape

    clf = svm.SVC(kernel="linear")
    clf.fit(train_embeddings, train_labels) 

    val_prediction = clf.predict(valid_embeddings)
    accuracy = np.sum(val_prediction == valid_labels) / num_valid_samples
    print(f"Validation accuracy: {accuracy}")

    boundary = clf.coef_.reshape(1, latent_dim).astype(np.float32)
    boundary = boundary / np.linalg.norm(boundary)
    np.savez(f"./lerp-results/boundaries/quality/{args.weights}_quality_boundary.npz", boundary=boundary)




if __name__ == "__main__":
    main()