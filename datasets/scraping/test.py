
import json
import re
import os

metadata = json.load(open("./data/metadata.json", "r"))

lines = []
for key, value in metadata.items():

    basename = os.path.basename(value['image-id'])
    channel = value['msr-key']
    
    if not basename.endswith(".msr"):
        continue

    protein_name = key.replace("-", "")
    protein_name = protein_name.replace("_", "")
    protein_name = protein_name.lower()

    sentence = f"Channel {channel} from filename '{basename}' is for the <mask> protein.".lower()
    sentence = sentence.replace("-", " ")
    sentence = sentence.replace("_", " ")

    sentence = re.sub(r'{[^}]*}*', '', sentence)

    lines.append(sentence)

with open("testing.txt", "w") as f:
    f.write("\n".join(lines))