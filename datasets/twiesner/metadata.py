
import json
import os, subprocess

CONVERSION = {
    "bassoon" : "Basson",
    "homer" : "Homer",
    "nkcc2" : "NKCC2",
    "rim" : "Rim",
    "psd95" : "PSD95"
}

def get_files(src : str) -> list[dict]:
    """
    Copies files from path

    :param config: A `dict` of the configuration file
    """
    _src = "valeria-s3:" + src
    args = [
        "rclone", "lsjson", "-R", _src
    ]
    print("[----] Running : {}".format(" ".join(args)))
    p = subprocess.check_output(args).decode("utf-8")
    output = json.loads(p)
    output = [
        o for o in output if not o["IsDir"]
    ]
    print("[----] Done!")
    return output

def get_protein_images(files : list[dict], protein : str) -> list[dict]:
    """
    Gets the files of a protein
    """
    images = []
    for file in files:
        if CONVERSION[protein] in file["Path"]:
            proteins = file["Path"].split("/")[0].split("-")
            chan_id = proteins.index(CONVERSION[protein])
            images.append({
                "chan-id" : chan_id, 
                "image-id" : file["Path"],
                "image-type" : "msr" if file["Path"].endswith("msr") else "tif",
            })
    return images

def main():

    metadata = json.load(open("../metadata.json", "r"))
    files = get_files("flclab-private/FLCDataset/twiesner/synaptic-protein-paper")    

    for PROTEIN in CONVERSION.keys():
        if not PROTEIN in metadata:
            metadata[PROTEIN] = []

        current = [m["image-id"] for m in metadata[PROTEIN]]

        for meta in get_protein_images(files, PROTEIN):
            meta["user-id"] = "twiesner"
            image_id = meta["image-id"]
            if not image_id in current:
                metadata[PROTEIN].append(meta)
            else:
                metadata[PROTEIN][current.index(image_id)] = meta
    
    for key, values in metadata.items():
        print(key, len(values))
    print("total", sum([len(value) for value in metadata.values()]))
    json.dump(metadata, open("../metadata.json", "w"), indent=4, sort_keys=True)

if __name__ == "__main__":

    main()