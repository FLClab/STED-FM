
import pandas
import json
import os
import matplotlib
import argparse
import glob

from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent, apply_formatter
from plottable.plots import circled_image # image
from matplotlib import pyplot

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, default='mae-lightning-small')
parser.add_argument("--opts", nargs="+", default=[], 
                    help="Additional configuration options")    
args = parser.parse_args()

# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"

FMT = "{:0.2f}"
NAMEFMT = {
    "Graph-connectivity": "Graph\nConnectivity",
    "KBET": "KBET",
    "LISI-batch": "LISI\nBatch",
    "Silhouette-batch": "Silhouette\nBatch",
    "LISI-label": "LISI\nLabel",
    "Leiden-ARI": "Leiden\nARI",
    "Leiden-NMI": "Leiden\nNMI",
    "Silhouette-label": "Silhouette\nLabel",
    "mAP-nonrep": "mAP\nNonrep",
    "batch-correction": "Batch\nCorrection",
    "bio-metrics": "Bio\nMetrics",
    "overall": "Overall"
}

def get_scalar_mapppable(col_data=None, norm_type=None):
    if norm_type == "minmax":
        vmin = col_data.min()
        vmax = col_data.max()
    if norm_type == "interquartile":
        # taken from plottable.cmap.normed_cmap
        num_stds = 2.5
        _median, _std = col_data.median(), col_data.std()
        vmin = _median - num_stds * _std
        vmax = _median + num_stds * _std
    else:
        vmin, vmax = 0, 1

    cmap = pyplot.get_cmap("Purples")
    norm = matplotlib.colors.Normalize(vmin, vmax)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def load_data(files, batch_effect):
    """
    Load the data from the files

    :params files: dict of files
    :params batch_effect: the name of the batch effect

    :returns: A ``pandas.DataFrame`` with the data
    """
    dfs = []
    for key, file in files.items():
        with open(file, "r") as f:
            data = json.load(f)[batch_effect]
            data["model"] = key
            dfs.append(data)
    return pandas.DataFrame(dfs)

def get_files():
    files = glob.glob(os.path.join("..", "results", f"{args.dataset}_{args.model}_*.json"))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for {args.dataset}_{args.model}_*.json")

    out = {}
    for file in files:
        for key in ["ImageNet", "HPA", "JUMP", "STED"]:
            if key.lower() in file.lower():
                out[key] = file
                break
    return out

def main():
    effects = [
        "geometric", "poisson", "gaussian-noise", "gaussian-blur", "mixed"
    ]

    files = get_files()
    for effect in effects:
        df = load_data(files, effect)
        df = df.set_index("model")

        # Create the table
        batch_correction_cols = ["Graph-connectivity", "KBET", "LISI-batch", "Silhouette-batch"]
        bio_metrics_cols = ["LISI-label", "Leiden-ARI", "Leiden-NMI", "Silhouette-label", "mAP-nonrep"]
        aggregate_cols = ["batch-correction", "bio-metrics", "overall"]

        df["batch-correction"] = df[batch_correction_cols].mean(axis=1)
        df["bio-metrics"] = df[bio_metrics_cols].mean(axis=1)

        # Weights defined in the paper
        df["overall"] = df["batch-correction"] * 0.4 + df["bio-metrics"] * 0.6

        df = df.sort_values(by="overall", ascending=False)
        df = df[batch_correction_cols + bio_metrics_cols + aggregate_cols]
        
        mappable = get_scalar_mapppable()
        cmap=mappable.to_rgba
        textprops = {"ha" : "center", "bbox": {"boxstyle": "circle", "pad": 0.35}}
        col_defs = (
            [
                ColumnDefinition(name="model", title="Model", textprops={"ha" : "left", "weight": "bold"}, width=1.25),
            ] + [
                ColumnDefinition(name=batch_correction_cols[0], title=NAMEFMT[batch_correction_cols[0]], group="Batch Correction", formatter="{:0.2f}", textprops=textprops, cmap=cmap)
            ] + [
                ColumnDefinition(name=col, title=NAMEFMT[col], group="Batch Correction", formatter="{:0.2f}", textprops=textprops, cmap=cmap)
                for col in batch_correction_cols[1:]
            ] + [
                ColumnDefinition(name=bio_metrics_cols[0], title=NAMEFMT[bio_metrics_cols[0]], group="Bio metrics", formatter="{:0.2f}", textprops=textprops, cmap=cmap)
            ] + [
                ColumnDefinition(name=col, title=NAMEFMT[col], group="Bio metrics", formatter="{:0.2f}", textprops=textprops, cmap=cmap)
                for col in bio_metrics_cols[1:]
            ] + [
                ColumnDefinition(name=aggregate_cols[0], title=NAMEFMT[aggregate_cols[0]], group="Aggregate score", formatter="{:0.2f}", textprops=textprops, cmap=cmap, border="left")
            ] + [
                ColumnDefinition(name=col, title=NAMEFMT[col], group="Aggregate score", formatter="{:0.2f}", textprops=textprops, cmap=cmap)
                for col in aggregate_cols[1:]
            ]
        )

        pyplot.rcParams["font.family"] = ["DejaVu Sans"]
        pyplot.rcParams["savefig.bbox"] = "tight"

        fig, ax = pyplot.subplots(figsize=(15, 6))
        table = Table(
            df,
            column_definitions=col_defs,
            row_dividers=True, footer_divider=True,
            ax=ax,
            textprops={"fontsize": 10},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},   
        ).autoset_fontcolors(colnames=batch_correction_cols + bio_metrics_cols + aggregate_cols)

        os.makedirs("tables", exist_ok=True)
        fig.savefig(os.path.join("tables", f"{args.dataset}_{args.model}_{effect}.png"), bbox_inches="tight", facecolor=ax.get_facecolor(), dpi=300)

if __name__ == "__main__":
    main()