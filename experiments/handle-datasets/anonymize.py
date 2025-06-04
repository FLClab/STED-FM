import numpy as np
import tarfile 
from tqdm import tqdm 
import argparse 
import io

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--dataset-path", type=str, default="/home-local/Frederic/Datasets/ALS/ALS_JM_Fred_unmixed")
args = parser.parse_args()

with tarfile.open(f"{args.dataset_path}/synaptic_development_dataset_{args.split}.tar", "r") as source_handle:
    with tarfile.open(f"{args.dataset_path}/synaptic_development_dataset_{args.split}_anonymized.tar", "w") as target_handle:

        members = source_handle.getmembers()
        for m, member in enumerate(tqdm(members, desc="Processing files...")):
            buffer = io.BytesIO()
            buffer.write(source_handle.extractfile(member).read())
            buffer.seek(0)
            data = np.load(buffer, allow_pickle=True)
            data = {key : values for key, values in data.items()}
            
            new_buffer = io.BytesIO()
            np.savez(
                new_buffer, 
                image=data["image"], 
                mask=data["mask"], 
                condition=data["condition"],
                div=data["div"], 
                dpi=data["dpi"], 
                min_value=data["min_value"], 
                max_value=data["max_value"])
            new_buffer.seek(0)
            name = str(m)
            tarinfo = tarfile.TarInfo(name=name)
            tarinfo.size = len(new_buffer.getbuffer())
            target_handle.addfile(tarinfo=tarinfo, fileobj=new_buffer)
            
    
        

if __name__=="__main__":
    pass