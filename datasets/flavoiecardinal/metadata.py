
import json
import os, subprocess

CONVERSION = {
    "camkii" : "CaMKII_Neuron",
    "f-actin" : "actin",
    "lifeact" : "LifeAct_Neuron",
    "psd95" : "PSD95_Neuron",
    "tubulin" : "tubulin"
}

def get_files(src):
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
    return output

def main():

    # Opens metadata and removes all previoius images from user
    metadata = json.load(open("../metadata.json", "r"))
    for key, values in metadata.items():
        deletevalues = []
        for i, value in enumerate(values):
            if value["user-id"] == "flavoiecardinal":
                deletevalues.append(i)
        for i in reversed(deletevalues):
            del metadata[key][i]       

    # Actin paper
    PROTEIN = "f-actin"
    if not PROTEIN in metadata:
        metadata[PROTEIN] = []
    current = [m["image-id"] for m in metadata[PROTEIN]]
    files = get_files("flclab-private/FLCDataset/flavoiecardinal/factin-dendrite-lavoiecardinal")
    for file in files:
        image_id = file["Path"]
        image_type = os.path.splitext(os.path.basename(image_id))[1][1:]
        meta = {
            "image-id" : os.path.join("flavoiecardinal/factin-dendrite-lavoiecardinal", image_id),
            "image-type" : image_type,
            "chan-id" : 0,
            "user-id" : "flavoiecardinal",
        }
        if not image_id in current:
            metadata[PROTEIN].append(meta)
        else:
            metadata[PROTEIN][current.index(image_id)] = meta
    
    # Optim paper
    for PROTEIN, folder_name in CONVERSION.items():
        if not PROTEIN in metadata:
            metadata[PROTEIN] = []        

        current = [m["image-id"] for m in metadata[PROTEIN]]
        files = get_files(f"flclab-private/FLCDataset/flavoiecardinal/optim-lavoiecardinal/{folder_name}")
        for file in files:
            image_id = file["Path"]
            basename = os.path.basename(image_id)
            if float(os.path.splitext(basename)[0].split("-")[-1]) > 0.6:
                image_type = os.path.splitext(os.path.basename(image_id))[-1][1:]
                meta = {
                    "image-id" : os.path.join(f"flavoiecardinal/optim-lavoiecardinal/{folder_name}", image_id),
                    "image-type" : image_type,
                    "chan-id" : "arr_0",
                    "user-id" : "flavoiecardinal",
                }
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