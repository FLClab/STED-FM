
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
    for key, values in metadata.items():
        deletevalues = []
        for i, value in enumerate(values):
            if value["user-id"] == "jchabbert":
                deletevalues.append(i)
        for i in reversed(deletevalues):
            del metadata[key][i]

    cleaned_data = json.load(open("cleaned-data.json", "r"))
    
    for PROTEIN, values in cleaned_data.items():
        
        if not PROTEIN in metadata:
            metadata[PROTEIN] = []
        current = [m["image-id"] for m in metadata[PROTEIN]]

        for value in values:
            chan_id = value["chan-id"]
            folder_names = value["images"]
            for folder_name in folder_names:
                files = get_files(os.path.join("flclab-private/FLCDataset", folder_name))
                for file in files:
                    image_id = file["Path"]
                    image_type = os.path.splitext(image_id)[-1][1:]
                    meta = {
                        "image-id" : os.path.join(folder_name, image_id),
                        "image-type" : image_type,
                        "chan-id" : chan_id,
                        "user-id" : "jchabbert",
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