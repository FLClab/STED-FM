
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
    PROTEIN = "f-actin"

    metadata = json.load(open("../metadata.json", "r"))
    if not PROTEIN in metadata:
        metadata[PROTEIN] = []

    for key, values in metadata.items():
        deletevalues = []
        for i, value in enumerate(values):
            if value["user-id"] == "flavoiecardinal":
                deletevalues.append(i)
        for i in reversed(deletevalues):
            del metadata[key][i]        

    current = [m["image-id"] for m in metadata[PROTEIN]]

    files = get_files("flclab-private/FLCDataset/flavoiecardinal")
    for file in files:
        image_id = file["Path"]
        meta = {
            "image-id" : os.path.join("flavoiecardinal", image_id),
            "image-type" : "msr" if image_id.endswith("msr") else "tif",
            "chan-id" : 0,
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