
import os, glob
import numpy
import tifffile

from tqdm.auto import tqdm
from stedfm.DEFAULTS import BASE_PATH

from neurofmdb.parsers.base_parser import get_parser_for_file

def make_figure3_dataset(dataset_path: str, outdir: str = f"{BASE_PATH}/evaluation-data/mRNAs/processed/figure3B"):
    
    MSRKEY = 'Alexa 594_STED {7}/Alexa 488_STED {12}/STAR RED_STED {7}'

    files = glob.glob(os.path.join(dataset_path, "**/*.msr"), recursive=True)

    training_files = [f for f in files if "Replicate 1" in f or "Replicate 2" in f]
    testing_files = [f for f in files if "Replicate 3" in f]

    for fold_name, files in zip(["train", "test"], [training_files, testing_files]):
        print(f"Processing {fold_name} files...")

        for f in tqdm(files, desc="Loading dataset"):
            parser = get_parser_for_file(f)
            data = parser.parse()
            try:
                image = data["data"][MSRKEY]
                metadata = data["other-metadata"]

                if "to_be_merged_keys" in data:
                    for keys in data["to_be_merged_keys"]:
                        joined_keys = "/".join(keys)
                        metadata[joined_keys] = [metadata[key] for key in keys]
                pass
            except KeyError:    
                print(f"Key {MSRKEY} not found in {f}")
                print(f"\tAvailable keys: {data['data'].keys()}")
                continue

            scale_ = 1.0
            if isinstance(metadata[MSRKEY], list):
                metadata_ = metadata[MSRKEY][0]
            else:
                metadata_ = metadata[MSRKEY]
            if metadata_["Pixels"]["PhysicalSizeXUnit"] == "µm":
                scale_ = 1.0
            elif metadata_["Pixels"]["PhysicalSizeXUnit"] == "nm":
                scale_ = 1e-3
            elif metadata_["Pixels"]["PhysicalSizeXUnit"] == "m":
                scale_ = 1e+6
            else:
                print(f"Unknown unit {metadata_['Pixels']['PhysicalSizeXUnit']}, assuming µm")
            
            dirname = os.path.dirname(f)
            dst = os.path.join(outdir, fold_name, os.path.basename(dirname))
            os.makedirs(dst, exist_ok=True)

            tifffile.imwrite(
                os.path.join(dst, os.path.basename(f).replace(".msr", ".tiff")),
                image.astype(numpy.uint16),
                resolution=(1. / (float(metadata_["Pixels"]["PhysicalSizeX"]) * scale_), 1. / (float(metadata_["Pixels"]["PhysicalSizeY"]) * scale_)),
                imagej=True,
                metadata = {"unit" : "um", "mode" : "composite"}
            )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default=f"{BASE_PATH}/evaluation-data/mRNAs/Raw Data_Stoldt et. al")
    parser.add_argument("--export-to-tiff", action="store_true")
    args = parser.parse_args()

    make_figure3_dataset(
        os.path.join(args.dataset_path, "Figure 3", "B"),
        f"{BASE_PATH}/evaluation-data/mRNAs/processed/figure3B"
    )
    make_figure3_dataset(
        os.path.join(args.dataset_path, "Figure 3", "D"),
        f"{BASE_PATH}/evaluation-data/mRNAs/processed/figure3D"
    )
    make_figure3_dataset(
        os.path.join(args.dataset_path, "Figure 3", "F"),
        f"{BASE_PATH}/evaluation-data/mRNAs/processed/figure3F"
    )

if __name__=="__main__":
    main()
    