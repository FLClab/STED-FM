
import json
import os, subprocess

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

    PROTEIN = "tom20"

    metadata = json.load(open("../metadata.json", "r"))
    for key, values in metadata.items():
        deletevalues = []
        for i, value in enumerate(values):
            if value["user-id"] == "oferguson":
                deletevalues.append(i)
        for i in reversed(deletevalues):
            del metadata[key][i]

    if not PROTEIN in metadata:
        metadata[PROTEIN] = []

    current = [m["image-id"] for m in metadata[PROTEIN]]

    # Inserts images
    files = get_files("flclab-private/FLCDataset/oferguson/Inserts_Images_and_Masks")
    for file in files:
        image_id = file["Path"]
        image_type = os.path.splitext(image_id)[-1][1:]
        meta = {
            "image-id" : os.path.join("oferguson/Inserts_Images_and_Masks", image_id),
            "image-type" : image_type,
            "chan-id" : 1,
            "user-id" : "oferguson",
        }
        if not image_id in current:
            metadata[PROTEIN].append(meta)
        else:
            metadata[PROTEIN][current.index(image_id)] = meta

    # Manual mitos images
    files = get_files("flclab-private/FLCDataset/oferguson/Manual_Mitos")
    for file in files:
        image_id = file["Path"]
        image_type = os.path.splitext(image_id)[-1][1:]
        meta = {
            "image-id" : os.path.join("oferguson/Manual_Mitos", image_id),
            "image-type" : image_type,
            "chan-id" : 2,
            "user-id" : "oferguson",
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