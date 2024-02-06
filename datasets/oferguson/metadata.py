
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
    metadata = json.load(open("../metadata.json", "r"))
    if not "tom20" in metadata:
        metadata["tom20"] = []

    current = [m["image-id"] for m in metadata["tom20"]]

    files = get_files("flclab-private/FLCDataset/oferguson")
    for file in files:
        image_id = file["Path"]
        meta = {
            "image-id" : image_id,
            "image-type" : "msr" if image_id.endswith("msr") else "tif",
            "chan-id" : 1,
            "user-id" : "oferguson",
        }
        if not image_id in current:
            metadata["tom20"].append(meta)
        else:
            metadata["tom20"][current.index(image_id)] = meta
    json.dump(metadata, open("../metadata.json", "w"), indent=4, sort_keys=True)

if __name__ == "__main__":

    main()