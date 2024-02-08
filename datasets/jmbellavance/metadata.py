import json
import os
import subprocess

FOLDERS = ["2023-01-12 (25) (cryodendrites)", "2023-02-15 (26.1) (dendrites 4c)", "2023-03-01 (26.2) (dendrites 4c Batch3)"]


def get_files(src: str):
    _src = "/Users/fredbeaupre/valeria-s3/flclab-private/" + src
    _src = "valeria-s3:" + src
    args = ["rclone", "lsjson", "-R", _src]
    print("[-----] Running: {}".format(" ".join(args)))
    p = subprocess.check_output(args).decode("utf-8")
    output = json.loads(p)
    output = [o for o in output if not o["IsDir"]]
    return output

def main():
    metadata = json.load(open("../metadata.json", "r"))
    for key, values in metadata.items():
        delete_values = []
        for i, value in enumerate(values):
            if value["user-id"] == "jmbellavance":
                delete_values.append(i)
        for i in reversed(delete_values):
            del metadata[key][i]

    for folder_name in FOLDERS:
        for PROTEIN in ["FUS", "psd95"]:
            channel_id = "STED_594 {6}" if PROTEIN == "FUS" else "STED_635P {6}"
            if not PROTEIN in metadata:
                metadata[PROTEIN] = []
            current = [m["image-id"] for m in metadata[PROTEIN]]
            files = get_files(f"flclab-private/FLCDataset/jmbellavance/ALS_FUS_and_PSD95/{folder_name}")
            for f in files:
                image_id = f["Path"]
                image_type = os.path.splitext(os.path.basename(image_id))[1][1:]
                meta = {
                    "image-id": os.path.join(f"jmbellavance/ALS_FUS_and_PSD95/{folder_name}", image_id),
                    "image-type": image_type,
                    "chan-id": channel_id,
                    "user-id": "jmbellavance"
                }
                if not image_id in current:
                    metadata[PROTEIN].append(meta)
                else:
                    metadata[PROTEIN][current.index(image_id)] = meta


    for key, values in metadata.items():
        print(key, len(values))
    print("total", sum([len(value) for value in metadata.values()]))
    json.dump(metadata, open("../metadata.json", "w"), indent=4, sort_keys=True)

if __name__=="__main__":
    main()