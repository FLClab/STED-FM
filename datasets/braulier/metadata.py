
import json
import os, subprocess

def get_files(src):
    """
    Copies files from path

    :param config: A `dict` of the configuration file
    """
    _src = "valeria-s3:" + src
    # _src = "/Users/fredbeaupre/valeria-s3/" + src
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

    USERID = "braulier"

    metadata = json.load(open("../metadata.json", "r"))
    for key, values in metadata.items():
        deletevalues = []
        for i, value in enumerate(values):
            if value["user-id"] == USERID:
                deletevalues.append(i)
        for i in reversed(deletevalues):
            del metadata[key][i]

    for PROTEIN in ["f-actin", "beta-camkii"]:
        chan_id = 0 if PROTEIN == "f-actin" else 1
        if not PROTEIN in metadata:
            metadata[PROTEIN] = []
        current = [m["image-id"] for m in metadata[PROTEIN]]
        files = get_files("flclab-private/FLCDataset/braulier/2-Color STED CaMKII-ACTIN")
        for file in files:
            image_id = file["Path"]
            image_type = os.path.splitext(image_id)[-1][1:]
            meta = {
                "image-id" : os.path.join("braulier/2-Color STED CaMKII-ACTIN", image_id),
                "image-type" : image_type,
                "chan-id" : chan_id,
                "user-id" : USERID,
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