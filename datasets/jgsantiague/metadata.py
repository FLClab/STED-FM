import json 
import os, subprocess
import argparse 

def get_files(src: str):
    _src = "/Users/fredbeaupre/valeria-s3/flclab-private/" + src
    args = ["rclone", "lsjson", "-R", _src]
    print("[-----] Running: {}".format(" ".join(args)))
    p = subprocess.check_output(args).decode("utf-8")
    output = json.loads(p)
    output = [o for o in output if not o["IsDir"]]
    return output

def main():
    # Opens metadata and removes all previous images from user
    metadata = json.load(open("../metadata.json", "r"))
    for key, values in metadata.items():
        delete_values = []
        for i, value in enumerate(values):
            if value["user-id"] == "jgsantiague":
                delete_values.append(i)
        for i in reversed(delete_values):
            del metadata[key][i]


    for folder_name in ["no_beads_2023-08-09", "no_beads_2023-09-23"]:
        for PROTEIN in ["bassoon", "psd95"]:
            channel_id = "STED 594 {12}" if PROTEIN == "bassoon" else "STED 635 {12}"
            if not PROTEIN in metadata:
                metadata[PROTEIN] = []
            current = [m["image-id"] for m in metadata[PROTEIN]]
            files = get_files(f"FLCDataset/jgsantiague/{folder_name}")
            for f in files:
                image_id = f["Path"]
                image_type = os.path.splitext(os.path.basename(image_id))[1][1:]
                meta = {
                    "image-id": os.path.join(f"jgsantiague/{folder_name}", image_id),
                    "image-type": image_type,
                    "chan-id": channel_id,
                    "user-id": "jgsantiague"
                }
                if not image_id in current:
                    metadata[PROTEIN].append(meta)
                else:
                    metadata[PROTEIN][current.index(image_id)] = meta

    for key, values in metadata.items():
        print(key, len(values))
    print("total", sum([len(values) for value in metadata.values()]))
    json.dump(metadata, open("../metadata.json", "w"), indent=4, sort_keys=True)





if __name__=="__main__":
    main()