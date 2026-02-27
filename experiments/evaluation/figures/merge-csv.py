
import pandas
import os 
import glob

from stedfm.DEFAULTS import DATASETS

DATASETS.optim = "SO"
DATASETS.neural_activity_states = "NAS"
DATASETS.peroxisome = "Px"
DATASETS.polymer_rings = "PR"
DATASETS.dl_sim = "DL-SIM"
DATASETS.bbbc026 = "BBBC026"
DATASETS.bbbc052 = "BBBC052"
DATASETS.bbbc053 = "BBBC053"
DATASETS.hpa_classification = "HPA-C"

MODEL = "mae-small"
datasets = [
    "optim",
    "neural-activity-states",
    "peroxisome",
    "polymer-rings",
    "dl-sim",
    "bbbc026",
    "bbbc052",
    "bbbc053",
    "hpa-classification"
]
modes = [
    "linear-probe",
    "finetuned",
]
samples = [
    "10",
    "25",
    "50",
    "100",
    "Full"
]

MODES = {
    "finetuned" : "FT",
    "linear-probe" : "LP",
}

def main():
    
    with pandas.ExcelWriter(os.path.join(".", "results", "raw-outputs", f"classification-experiments-merged.xlsx")) as writer:
        for dataset in datasets:
            for mode in modes:
                df = pandas.read_csv(os.path.join(".", "results", "raw-outputs", f"{MODEL}_{dataset}_{mode}-small-dataset.csv"), index_col=0)
                df.to_excel(writer, sheet_name=f"{DATASETS[dataset]}-{MODES[mode]}")

                file = glob.glob(os.path.join(".", "results", "raw-outputs", f"{MODEL}_{dataset}_{mode}_F*-small-dataset-stats.csv"))[0]
                p_value = float(file.split("_F")[1].split("-small")[0])
                stats_df = pandas.read_csv(file, index_col=0)
                stats_df.to_excel(writer, sheet_name=f"{DATASETS[dataset]}-{MODES[mode]} (p={p_value:0.4e})")

    with pandas.ExcelWriter(os.path.join(".", "results", "raw-outputs", f"classification-experiments-ranked.xlsx")) as writer:
        for sample in samples:
            for mode in modes:
                file = glob.glob(os.path.join(".", "results", "raw-outputs", f"{MODEL}_STED_{mode}_H*_ranking_stats_sample-{sample}.csv"))[0]
                p_value = float(file.split("_H")[1].split("_ranking")[0])
                df = pandas.read_csv(file, index_col=0)
                df.to_excel(writer, sheet_name=f"Sample-{sample}-{MODES[mode]} (p={p_value:0.4e})")

if __name__ == "__main__":

    main()