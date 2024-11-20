
import numpy

from matplotlib import pyplot
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import RFE, SelectKBest, r_regression, f_regression, mutual_info_regression, SequentialFeatureSelector

import sys
sys.path.insert(0, "..")
from utils import savefig

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42,
                help="Random seed")     
parser.add_argument("--model", type=str, default="resnet18",
                    help="model model to load")
parser.add_argument("--weights", type=str, default=None,
                    help="Backbone model to load")
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")
parser.add_argument("--dry-run", action="store_true",
                    help="Activates dryrun")        
args = parser.parse_args()

# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"
# Ensure backbone weights are provided if necessary
if args.weights in (None, "null", "None", "none"):
    args.weights = None

def main():

    data = numpy.load(f"features-{args.weights}.npz")

    X_train, y_train = data["X_train"], data["y_train"]
    X_valid, y_valid = data["X_valid"], data["y_valid"]
    X_test, y_test = data["X_test"], data["y_test"]

    # sfs = SequentialFeatureSelector(
    #     Ridge(),
    #     n_features_to_select="auto",
    #     tol=0.01,
    #     direction="forward",
    #     cv=3,
    #     n_jobs=-1
    # )
    # sfs.fit(X_train, y_train)

    # model = Ridge()
    # model.fit(X_train[:, sfs.get_support()], y_train)
    # print(model.score(X_valid[:, sfs.get_support()], y_valid))

    # mask = numpy.invert(sfs.get_support())

    # X_train = X_train[:, mask]
    # X_valid = X_valid[:, mask]
    # X_test = X_test[:, mask]

    feature_ids = numpy.arange(X_train.shape[1])
    for _ in range(1):

        print(X_train.shape[-1])

        f_statistic, p_values = f_regression(X_train, y_train)
        # argsorted = numpy.argsort(f_statistic)
        # print(numpy.max(f_statistic), numpy.min(f_statistic))

        # # Keep 90% of the features
        mask = p_values > 0.05
        # mask = argsorted[:int(0.9 * X_train.shape[1])]
        X_train = X_train[:, mask]
        X_valid = X_valid[:, mask]
        X_test = X_test[:, mask]
        feature_ids = feature_ids[mask]

        # # Fit the model
        # model = Ridge()
        # model.fit(X_train, y_train)
        # score = model.score(X_valid, y_valid)
        # print(score)
        # if (score < 0.05) or (len(mask) <= 5):
        #     break
    print(len(feature_ids))

    # print(feature_ids)
    # numpy.save(f"feature-ids-{args.weights}.npy", feature_ids)

    # Fit the model
    model = Ridge()
    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))
    pred = model.predict(X_valid)

    fig, ax = pyplot.subplots(figsize=(3, 3))
    ax.scatter(y_valid, pred, alpha=0.5)
    ax.set(
        xlabel="True quality score",
        ylabel="Predicted quality score",
        ylim=(0, 1),
        xlim=(0, 1),
    )
    savefig(fig, "tmp/quality", save_white=True)

if __name__ == "__main__":

    main()