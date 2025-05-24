import tarfile 
import numpy as np
import io 
from tqdm import tqdm 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--path", type=str, default="/home-local/Frederic/Datasets/ALS/ALS_JM_Fred_unmixed")
args = parser.parse_args()

if __name__=="__main__":
    with tarfile.open(f"{args.path}/PLKO-262-PSD95-{args.split}.tar", "r") as source:
        members = source.getmembers()

        print(f"Total members in source: {len(members)}")
        
        with tarfile.open(f"{args.path}/synaptic_development_dataset_{args.split}.tar", "w") as destination:
            dest_members = 0
            for member in tqdm(members):
                file_content = source.extractfile(member).read()
                buffer = io.BytesIO(file_content)
                data = np.load(buffer, allow_pickle=True)
                div, dpi = data["div"], data["dpi"] 
                if "5" in str(div) and "4" in str(dpi):
                    continue 
                else:
                    dest_members += 1
                    dest_buffer = io.BytesIO()
                    np.savez(
                        dest_buffer,    
                        image=data["image"], 
                        mask=data["mask"], 
                        condition=data["condition"], 
                        div=data["div"], 
                        dpi=data["dpi"], 
                        min_value=data["min_value"], 
                        max_value=data["max_value"]
                    )
                    dest_buffer.seek(0)
                    tarinfo = tarfile.TarInfo(name=member.name)
                    tarinfo.size = len(dest_buffer.getbuffer())
                    destination.addfile(tarinfo=tarinfo, fileobj=dest_buffer)

        print(f"Total members in destination: {dest_members}")

                
