import numpy as np
import matplotlib.pyplot as plt
from typing import List
import datasets
from torch.utils.data import DataLoader


def get_STED_dataset(transform, path: str):
    dataset = datasets.TarFLCDataset(tar_path=path, transform=transform)
    dataloader = DataLoader(data=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader

def get_CTC_dataset(transform, path: str):
    dataset = datasets.CTCDataset(h5file=path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader

def get_synaptic_proteins_dataset(
        path: str,
        class_ids: List,
        batch_size: int = 256,
        validation_size: float = 0.10,
        class_type: str = "protein",
        n_channels: int = 1,
        h5size: int = 67436,
) -> DataLoader:
    np.random.seed(42) # For reproducibility and to always get same test set when we evaluate
    indices = np.arange(0, h5size, 1)
    np.random.shuffle(indices) 
    split = int( (h5size) * (1 - validation_size))
    temp_indices = indices[:split]
    test_indices = indices[split:]

    val_split = int(( (temp_indices.shape[0]) * (1 - validation_size)))
    train_indices = temp_indices[:val_split]
    val_indices = temp_indices[val_split:]

    train_dataset = datasets.ProteinDataset(
        h5file=f"{path}/theresa_proteins.hdf5",
        class_ids=None,
        class_type=class_type,
        n_channels=n_channels,
        indices=train_indices
    )
    valid_dataset = datasets.ProteinDataset(
        h5file=f"{path}/theresa_proteins.hdf5",
        class_ids=None,
        class_type=class_type,
        n_channels=n_channels,
        indices=val_indices
    )
    test_dataset = datasets.ProteinDataset(
            h5file=f"{path}/theresa_proteins.hdf5",
            class_ids=None,
            class_type=class_type,
            n_channels=n_channels,
            indices=test_indices
    )
    print(f"Training size: {len(train_dataset)}")
    print(f"Validation size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
    )
    return train_loader, valid_loader, test_loader

def get_optim_dataset(path: str, **kwargs):
    dataset = datasets.OptimDataset(
        data_folder=path,
        num_samples={'actin': None, 'tubulin': None, 'CaMKII_Neuron': None, "PSD95_Neuron": None},
        apply_filter=True,
        classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
        **kwargs
    )
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader

def get_factin_rings_fibers_dataset(path: str, **kwargs):
    dataset = datasets.CreateFactinRingsFibersDataset(data_folder=path, classes=["rings", "fibers"], **kwargs)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataset

def get_factin_block_glugly_dataset(path: str, **kwargs):
    dataset = datasets.CreateFactinRingsFibersDataset(data_folder=path, classes=["Block", "GLU-GLY"], **kwargs)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataset

def get_dataset(name, path, **kwargs):
    if name == "optim":
        return get_optim_dataset(path=path, transform=kwargs['transform'])
    elif name == "synaptic-proteins":
        return get_synaptic_proteins_dataset(path=path, n_channels=kwargs['n_channels'], transform=kwargs['transform'])
    elif name == "factin-rings-fibers":
        return get_factin_rings_fibers_dataset(path=path, transform=kwargs['transform'])
    elif name == "factin-block-glugly":
        return get_factin_block_glugly_dataset(path=path, transform=kwargs['transform'])
    else:
        raise NotImplementedError(f"`{name}` dataset is not supported.")




    

