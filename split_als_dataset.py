import tarfile 
import numpy as np 
import io
from tqdm import tqdm
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=str, default=None)
args = parser.parse_args()

PATH = "/home-local/Frederic/Datasets/ALS/catalog"

if __name__ == "__main__":
    with tarfile.open(f"{PATH}/train_ALS.tar", "r") as tar:
        members = tar.getmembers()
        N = len(members)
        print(f"Total: {N}")
        train_members = np.random.choice(members, size=int(0.5 * N), replace=False)
        val_members = [item for item in members if item not in train_members]
        test_members = np.random.choice(val_members, size=int(0.5 * len(val_members)), replace=False)
        
        val_members = [item for item in val_members if item not in test_members]

        assert len(train_members) + len(val_members) + len(test_members) == N
        print(f"Train: {len(train_members)}, Val: {len(val_members)}, Test: {len(test_members)}")
        assert not any(item in test_members for item in val_members)
        assert not any(item in val_members for item in train_members)

        num_train = 0
        with tarfile.open(f"{PATH}/PLKO_train.tar", "w") as plko_train_tar:
            for member in tqdm(train_members, total=len(train_members), desc="Train..."):
                # Read the file content first
                file_content = tar.extractfile(member).read()
                buffer = io.BytesIO(file_content)
                data = np.load(buffer, allow_pickle=True)
                metadata = data["metadata"].item()
                batch = metadata["batch_id"]
                if args.batch is not None and batch != args.batch:
                    continue
                assert batch == args.batch
                num_train += 1
                # Reset buffer position and create new buffer with original content
                buffer = io.BytesIO(file_content)
                tarinfo = tarfile.TarInfo(name=member.name)
                tarinfo.size = len(file_content)
                plko_train_tar.addfile(tarinfo=tarinfo, fileobj=buffer)
        print(f"Num train: {num_train}")

        num_valid = 0
        with tarfile.open(f"{PATH}/PLKO_valid.tar", "w") as plko_valid_tar:
            for member in tqdm(val_members, total=len(val_members), desc="Valid..."):
                file_content = tar.extractfile(member).read()
                buffer = io.BytesIO(file_content)
                data = np.load(buffer, allow_pickle=True)
                metadata = data["metadata"].item()
                batch = metadata["batch_id"]
                if args.batch is not None and batch != args.batch:
                    continue
                assert batch == args.batch
                num_valid += 1
                buffer = io.BytesIO(file_content)
                tarinfo = tarfile.TarInfo(name=member.name)
                tarinfo.size = len(file_content)
                plko_valid_tar.addfile(tarinfo=tarinfo, fileobj=buffer)
        print(f"Num valid: {num_valid}")

        num_test = 0    
        with tarfile.open(f"{PATH}/PLKO_test.tar", "w") as plko_test_tar:
            for member in tqdm(test_members, total=len(test_members), desc="Test..."):
                file_content = tar.extractfile(member).read()
                buffer = io.BytesIO(file_content)
                data = np.load(buffer, allow_pickle=True)
                metadata = data["metadata"].item()
                batch = metadata["batch_id"]
                if args.batch is not None and batch != args.batch:
                    continue
                assert batch == args.batch
                num_test += 1
                buffer = io.BytesIO(file_content)
                tarinfo = tarfile.TarInfo(name=member.name)
                tarinfo.size = len(file_content)
                plko_test_tar.addfile(tarinfo=tarinfo, fileobj=buffer)
        print(f"Num test: {num_test}")


