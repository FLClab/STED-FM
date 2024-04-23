import numpy as np
import matplotlib.pyplot as plt
from typing import List
import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import Dataset
import random

# class BalancedSampler(Sampler):
#     def __init__(self, dataset: torch.utils.data.Dataset, num_samples):
#         self.dataset = dataset
#         self.num_samples = num_samples
#         self.ind0 = np.argwhere(self.dataset.labels == 0)
#         self.ind0 = np.random.choice(self.ind0.ravel(), size=num_samples//2, replace=False)
#         print(f"Indices 0 shape: {self.ind0.shape}")
#         self.ind1 = np.argwhere(self.dataset.labels == 1)
#         self.ind1 = np.random.choice(self.ind1.ravel(), size=num_samples//2, replace=False)
#         print(f"Indices 1 shape: {self.ind1.shape}")
        
#     def __len__(self):
#         return self.num_samples // 2 + self.num_samples // 2
    
#     def __iter__(self):
#         ids = np.concatenate([self.ind0.ravel(), self.ind1.ravel()])
#         print(f"All indices shape: {ids.shape}")
#         random.shuffle(ids)
#         return iter(ids)

class BalancedSampler(Sampler):
    def __init__(self, dataset: Dataset, fewshot_pct: float = 0.01, num_classes: int = 4) -> None:
        self.dataset = dataset
        self.fewshot_pct = fewshot_pct
        self.full_size = len(dataset)
        self.num_classes = num_classes
        self.target_size = int(fewshot_pct * self.full_size)
        self.num_samples = self.target_size // num_classes
        self.indices = [] 
        for i in range(self.num_classes):
            inds = np.argwhere(np.array(self.dataset.labels) == i)
            inds = np.random.choice(inds.ravel(), size=self.num_samples, replace=False)
            self.indices.append(inds)

    def __len__(self):
        return self.indices.shape[0]
    
    def __iter__(self):
        ids = np.concatenate([ids.ravel() for ids in self.indices]).astype('int')
        print(np.unique(np.array(self.dataset.labels)[ids], return_counts=True))
        random.shuffle(ids)
        print(np.unique(np.array(self.dataset.labels)[ids], return_counts=True))
        return iter(ids)


def get_STED_dataset(transform, path: str):
    dataset = datasets.TarFLCDataset(tar_path=path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader

def get_CTC_dataset(transform, path: str):
    dataset = datasets.CTCDataset(h5file=path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader

def get_synaptic_proteins_dataset(
    path: str,
    transform,
    class_ids: List = None,
    batch_size: int = 256,
    validation_size: float = 0.10,
    class_type: str = 'protein',
    n_channels: int = 1,
    training: bool = False,
    fewshot_pct: float = 0.0,
):
    train_dataset = datasets.ProteinDataset(
        h5file=f"{path}/train.hdf5",
        class_ids=None,
        class_type=class_type,
        n_channels=n_channels,
        transform=transform,
    )
    validation_dataset = datasets.ProteinDataset(
        h5file=f"{path}/validation.hdf5",
        class_ids=None,
        class_type=class_type,
        n_channels=n_channels,
        transform=transform,
    )
    test_dataset = datasets.ProteinDataset(
        h5file=f"{path}/test.hdf5",
        class_ids=None,
        class_type=class_type,
        n_channels=n_channels,
        transform=transform,
    )

    print(f"Training size: {len(train_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    if fewshot_pct == 1.0:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=6,
        )
    else:
        print(f"Loading balanced training set with {fewshot_pct * 100}% of labels")
        sampler = BalancedSampler(dataset=train_dataset, fewshot_pct=fewshot_pct, num_classes=4)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=6,
            sampler=sampler
        )
    valid_loader = DataLoader(
        dataset=validation_dataset,
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
    if training: 
        return train_loader, valid_loader, test_loader
    else:
        return test_loader



def get_synaptic_proteins_dataset_old(
        path: str,
        transform,
        class_ids: List = None,
        batch_size: int = 256,
        validation_size: float = 0.10,
        class_type: str = "protein",
        n_channels: int = 1,
        h5size: int = 67436,
        training: bool = False,
        fewshot_pct: float = 0.0,
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

    if fewshot_pct == 0.0:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=6,
        )
    else:
        print(f"Loading balanced training set with {fewshot_pct * 100}% of labels")
        sampler = BalancedSampler(dataset=train_dataset, fewshot_pct=fewshot_pct, num_classes=4)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=6,
            sampler=sampler
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
    if training: 
        return train_loader, valid_loader, test_loader
    else:
        return test_loader

def get_optim_dataset(path: str, training: bool = False, batch_size=256, **kwargs):
    if training: # Disregards the provided path
        train_dataset = datasets.OptimDataset(
            data_folder="/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/optim_train",
            num_samples={'actin': None, 'tubulin': None, 'CaMKII_Neuron': None, "PSD95_Neuron": None},
            apply_filter=True,
            classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
            **kwargs
        )
        valid_dataset = datasets.OptimDataset(
            data_folder="/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/optim_valid",
            num_samples={'actin': None, 'tubulin': None, 'CaMKII_Neuron': None, "PSD95_Neuron": None},
            apply_filter=True,
            classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
            **kwargs
        )
        test_dataset = datasets.OptimDataset(
            data_folder="/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/optim-data",
            num_samples={'actin': None, 'tubulin': None, 'CaMKII_Neuron': None, "PSD95_Neuron": None},
            apply_filter=True,
            classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
            **kwargs
        )
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Valid dataset size: {len(valid_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
        return train_loader, valid_loader, test_loader
    else:
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
        return get_optim_dataset(
            path="/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/optim-data", 
            n_channels=kwargs['n_channels'],
            transform=kwargs['transform'],
            training=kwargs['training'],
            batch_size=kwargs['batch_size']
            )
    elif name == "synaptic-proteins":
        return get_synaptic_proteins_dataset(
            path="/home/frbea320/projects/def-flavielc/frbea320/flc-dataset/experiments/Datasets/FLCDataset/TheresaProteins", 
            n_channels=kwargs['n_channels'], 
            transform=kwargs['transform'],
            training=kwargs['training'],
            batch_size=kwargs['batch_size'],
            fewshot_pct=kwargs['fewshot_pct']
            )
    elif name == "factin-rings-fibers":
        return get_factin_rings_fibers_dataset(path=path, transform=kwargs['transform'])
    elif name == "factin-block-glugly":
        return get_factin_block_glugly_dataset(path=path, transform=kwargs['transform'])
    else:
        raise NotImplementedError(f"`{name}` dataset is not supported.")




    

