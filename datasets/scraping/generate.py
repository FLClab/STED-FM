
import os
import json
import re

from sklearn.model_selection import train_test_split

def main():

    lines = []
    protein_names = set()
    metadata = json.load(open("../metadata.json", "r"))
    for key, values in metadata.items():
        for value in values:
            basename = os.path.basename(value['image-id'])
            
            if not basename.endswith(".msr"):
                continue

            protein_name = key.replace("-", "")
            protein_name = protein_name.replace("_", "")
            protein_name = protein_name.lower()
            protein_names.add(protein_name)

            sentence = f"Channel {value['chan-id']} from filename '{basename}' is for the {protein_name} protein.".lower()
            sentence = sentence.replace("-", " ")
            sentence = sentence.replace("_", " ")

            sentence = re.sub(r'{[^}]*}*', '', sentence)

            lines.append(sentence)

    X_train, X_test = train_test_split(lines, test_size=0.2, random_state=42, shuffle=True)
    print(len(X_train), len(X_test))
    with open("training.txt", "w") as f:
        f.write("\n".join(X_train))
    with open("validation.txt", "w") as f:
        f.write("\n".join(X_test))

    print(protein_names)
    

if __name__ == "__main__":

    main()