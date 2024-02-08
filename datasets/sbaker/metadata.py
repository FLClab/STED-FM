import json 
import os 
import subprocess

FOLDERS = ["2023.05.30.P6FinalP7", "2023.06.21.P7final", "2023.07.13_4_color_session", "2023.07.20_p8", "2023.09.23", "2023.10.05"]

def get_files(src: str):
    _src = "/Users/fredbeaupre/valeria-s3/flclab-private/" + src
    args = ["rclone", "lsjson", "-R", _src]
    print("[-----] Running: {}".format(" ".join(args)))
    p = subprocess.check_output(args).decode("utf-8")
    output = json.loads(p)
    output = [o for o in output if not o["IsDir"]]
    return output

def main():
    metadata = json.load(open("../metadata.json", "r"))
    # for key, values in metadata.items():
    #     delete_values = []
    #     for i, value in enumerate(values):
    #         if value["user-id"] == "sbaker":
    #             delete_values.append(i)
    #     for i in reversed(delete_values):
    #         del metadata[key][i]

    # for folder_name in FOLDERS:
    #     for PROTEIN in ["vglut2", "psd95"]:
    #         channel_id = "STED 594 {6}" if PROTEIN == "vglut2" else "STED 635P {6}"
    #         if not PROTEIN in metadata:
    #             metadata[PROTEIN] = []
    #         current = [m["image-id"] for m in metadata[PROTEIN]]
    #         files = get_files(f"FLCDataset/sbaker/{folder_name}")
    #         for f in files:
    #             image_id = f["Path"]
    #             image_type = os.path.splitext(os.path.basename(image_id))[1][1:]
    #             meta = {
    #                 "image-id": os.path.join(f"sbaker/{folder_name}", image_id),
    #                 "image-type": image_type,
    #                 "chan-id": channel_id,
    #                 "user-id": "sbaker"
    #             }
    #             if not image_id in current:
    #                 metadata[PROTEIN].append(meta)
    #             else:
    #                 metadata[PROTEIN][current.index(image_id)] = meta

    for key, values in metadata.items():
        print(key, len(values))
    print("total", sum([len(value) for value in metadata.values()]))
    json.dump(metadata, open("../metadata.json", "w"), indent=4, sort_keys=True)

if __name__=="__main__":
    main()