import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Iterable, Callable
import datasets
from torch.utils.data import DataLoader, Sampler, Dataset
import torchvision.transforms as T
import random
import os
from DEFAULTS import BASE_PATH

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
            inds = np.random.choice(inds.ravel(), size=self.num_samples, replace=True)
            self.indices.append(inds)

    def __len__(self):
        return self.indices.shape[0]
    
    def __iter__(self):
        ids = np.concatenate([ids.ravel() for ids in self.indices]).astype('int')
        print(np.unique(np.array(self.dataset.labels)[ids], return_counts=True))
        random.shuffle(ids)
        print(np.unique(np.array(self.dataset.labels)[ids], return_counts=True))
        return iter(ids)
    
class UltraSmallSampler(Sampler):
    def __init__(self, dataset: Dataset, num_per_class: int = 400, num_classes: int = 4) -> None:
        self.dataset = dataset
        self.dataset_size = num_per_class * num_classes 
        self.indices = []
        for i in range(num_classes):
            inds = np.argwhere(np.array(self.dataset.labels) == i)
            inds = np.random.choice(inds.ravel(), size=num_per_class, replace=True)
            self.indices.append(inds)

    def __len__(self) -> int:
        return self.dataset_size
    
    def __iter__(self) -> Iterable:
        ids = np.concatenate([ids.ravel() for ids in self.indices]).astype(np.int64)
        random.shuffle(ids)
        return iter(ids)


from DEFAULTS import BASE_PATH

def get_JUMP_dataset(transform: Callable, path: str):
    dataset = datasets.TarJUMPDataset(tar_path=path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader


def get_STED_dataset(transform, path: str):
    dataset = datasets.TarFLCDataset(tar_path=path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader

def get_CTC_dataset(transform, path: str):
    dataset = datasets.CTCDataset(h5file=path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataloader

def get_neural_activity_states(
        path:str,
        transform: Callable,
        batch_size: int = 256,
        n_channels: int = 1,
        num_samples: int = None,
        protein_id: int = 3,
        **kwargs
):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ])
    train_dataset = datasets.NeuralActivityStates(
        tarpath=f"{path}/NAS_PSD95_train_v2.tar",
        transform=transform,
        n_channels=n_channels,
        num_samples=num_samples,
        num_classes=4,
        balance=kwargs.get("balance", True)
    )
    validation_dataset = datasets.NeuralActivityStates(
        tarpath=f"{path}/NAS_PSD95_valid_v2.tar",
        transform=transform,
        n_channels=n_channels,
        num_samples=None,
        num_classes=4,
        balance=kwargs.get("balance", True)
    )
    test_dataset = datasets.NeuralActivityStates(
        tarpath=f"{path}/NAS_PSD95_test_v2.tar",
        transform=transform,
        n_channels=n_channels,
        num_samples=None,
        num_classes=4,
        balance=kwargs.get("balance", True)
    )
    print("\n=== NAS dataset ===")
    print(np.unique(train_dataset.labels, return_counts=True))
    print(np.unique(validation_dataset.labels, return_counts=True))
    print(np.unique(test_dataset.labels, return_counts=True))

    print(f"Training size: {len(train_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    print(f"Test size: {len(test_dataset)}\n")
    print("======\n")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
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
    return train_loader, valid_loader, test_loader

def get_synaptic_proteins_dataset(
    path: str,
    transform,
    class_ids: List = None,
    batch_size: int = 256,
    class_type: str = 'proteins',
    n_channels: int = 1,
    num_samples: int = None,
):
    train_dataset = datasets.ProteinDataset(
        h5file=f"{path}/train_v2.hdf5",
        class_ids=None,
        class_type=class_type,
        n_channels=n_channels,
        transform=transform,
        num_samples=num_samples
    )
    validation_dataset = datasets.ProteinDataset(
        h5file=f"{path}/valid_v2.hdf5",
        class_ids=None,
        class_type=class_type,
        n_channels=n_channels,
        transform=transform,
    )
    test_dataset = datasets.ProteinDataset(
        h5file=f"{path}/test_v2.hdf5",
        class_ids=None,
        class_type=class_type,
        n_channels=n_channels,
        transform=transform,
    )

    print(np.unique(train_dataset.labels, return_counts=True))
    print(np.unique(validation_dataset.labels, return_counts=True))
    print(np.unique(test_dataset.labels, return_counts=True))

    print(f"Training size: {len(train_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    print(f"Test size: {len(test_dataset)}\n")
    ### Keeping code below if we want to revert back to sampling based on % of labels
    # if fewshot_pct == 1.0:
    #     train_loader = DataLoader(
    #         dataset=train_dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         drop_last=False,
    #         num_workers=6,
    #     )
    # else:
    #     print(f"Loading balanced training set with {fewshot_pct * 100}% of labels")
    #     sampler = BalancedSampler(dataset=train_dataset, fewshot_pct=fewshot_pct, num_classes=4)
    #     train_loader = DataLoader(
    #         dataset=train_dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         drop_last=False,
    #         num_workers=6,
    #         sampler=sampler
    #     )


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
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
    return train_loader, valid_loader, test_loader
  

def get_optim_dataset(path: str, training: bool = False, batch_size=256, num_samples=None, *args, **kwargs):
    # print(f"TESTING --> {num_samples} samples per class")
    samples_dict = {
        "actin": num_samples,
        "tubulin": num_samples,
        "CaMKII_Neuron": num_samples,
        "PSD95_Neuron": num_samples
    }

    # if training: # Disregards the provided path
    print(os.path.join(BASE_PATH, "evaluation-data", "optim_train"))
    
    train_dataset = datasets.OptimDataset(
        data_folder=os.path.join(BASE_PATH, "evaluation-data", "optim_train"),
        num_samples=samples_dict,
        apply_filter=True,
        classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
        **kwargs
    )
    valid_dataset = datasets.OptimDataset(
        data_folder=os.path.join(BASE_PATH, "evaluation-data", "optim_valid"),
        num_samples={'actin': None, 'tubulin': None, 'CaMKII_Neuron': None, "PSD95_Neuron": None},
        apply_filter=True,
        classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
        **kwargs
    )
    test_dataset = datasets.OptimDataset(
        data_folder=os.path.join(BASE_PATH, "evaluation-data", "optim-data"),
        num_samples={'actin': None, 'tubulin': None, 'CaMKII_Neuron': None, "PSD95_Neuron": None},
        apply_filter=True,
        classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'],
        **kwargs
    )
    print(f"Train dataset size: {len(train_dataset)} --> {np.unique(train_dataset.labels, return_counts=True)}")
    print(f"Valid dataset size: {len(valid_dataset)} --> {np.unique(valid_dataset.labels, return_counts=True)}")
    print(f"Test dataset size: {len(test_dataset)} --> {np.unique(test_dataset.labels, return_counts=True)}")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    return train_loader, valid_loader, test_loader

def get_factin_rings_fibers_dataset(path: str, **kwargs):
    dataset = datasets.CreateFactinRingsFibersDataset(data_folder=path, classes=["rings", "fibers"], **kwargs)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataset

def get_factin_block_glugly_dataset(path: str, **kwargs):
    dataset = datasets.CreateFactinRingsFibersDataset(data_folder=path, classes=["Block", "GLU-GLY"], **kwargs)
    dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=6)
    return dataset

def get_peroxisome_dataset(path: str, num_samples: int, batch_size: int = 128, **kwargs):

    training_dataset = datasets.PeroxisomeDataset(source=os.path.join(path, "peroxisome-training.txt"), num_samples=num_samples, **kwargs)
    validation_dataset = datasets.PeroxisomeDataset(source=os.path.join(path, "peroxisome-validation.txt"), num_samples=None, **kwargs)
    testing_dataset = datasets.PeroxisomeDataset(source=os.path.join(path, "peroxisome-testing.txt"), num_samples=None, **kwargs)

    print(f"Training size: {len(training_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    print(f"Test size: {len(testing_dataset)}")

    # statistics = []
    # for img, _ in training_dataset:
    #     statistics.append(np.mean(img.numpy()))
    # print(f"Training mean: {np.mean(statistics)}")
    # statistics = []
    # for img, _ in training_dataset:
    #     statistics.append(np.std(img.numpy()))
    # print(f"Training std: {np.std(statistics)}")

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    valid_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    return train_loader, valid_loader, test_loader

def get_polymer_rings_dataset(path: str, num_samples: int, batch_size: int = 128, **kwargs):

    training_dataset = datasets.PolymerRingsDataset(source=os.path.join(path, "polymer-rings-training.txt"), num_samples=num_samples, **kwargs)
    validation_dataset = datasets.PolymerRingsDataset(source=os.path.join(path, "polymer-rings-validation.txt"), num_samples=None, **kwargs)
    testing_dataset = datasets.PolymerRingsDataset(source=os.path.join(path, "polymer-rings-testing.txt"), num_samples=None, **kwargs)

    print(f"Training size: {len(training_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    print(f"Test size: {len(testing_dataset)}")

    # statistics = []
    # for img, _ in training_dataset:
    #     statistics.append(np.mean(img.numpy()))
    # print(f"Training mean: {np.mean(statistics)}")
    # statistics = []
    # for img, _ in training_dataset:
    #     statistics.append(np.std(img.numpy()))
    # print(f"Training std: {np.std(statistics)}")

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    valid_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    return train_loader, valid_loader, test_loader

def get_dl_sim_dataset(path: str, num_samples: int, batch_size: int = 128, **kwargs):

    training_dataset = datasets.DLSIMDataset(source=os.path.join(path, "DL-SIM-training.txt"), num_samples=num_samples, **kwargs)
    validation_dataset = datasets.DLSIMDataset(source=os.path.join(path, "DL-SIM-validation.txt"), num_samples=None, **kwargs)
    testing_dataset = datasets.DLSIMDataset(source=os.path.join(path, "DL-SIM-testing.txt"), num_samples=None, **kwargs)

    print(f"Training size: {len(training_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    print(f"Test size: {len(testing_dataset)}")

    # statistics = []
    # for img, _ in training_dataset:
    #     statistics.append(np.mean(img.numpy()))
    # print(f"Training mean: {np.mean(statistics)}")
    # statistics = []
    # for img, _ in training_dataset:
    #     statistics.append(np.std(img.numpy()))
    # print(f"Training std: {np.std(statistics)}")

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    valid_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)
    return train_loader, valid_loader, test_loader

def get_resolution_dataset(path: str, batch_size: int = 128, **kwargs):
    training_dataset = datasets.ResolutionDataset(path=os.path.join(path, "training.hdf5"), **kwargs)
    validation_dataset = datasets.ResolutionDataset(path=os.path.join(path, "validation.hdf5"), **kwargs)
    testing_dataset = datasets.ResolutionDataset(path=os.path.join(path, "testing.hdf5"), **kwargs)

    print(f"Training size: {len(training_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    print(f"Test size: {len(testing_dataset)}")

    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    valid_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

    return train_loader, valid_loader, test_loader

def get_dataset(name, path=None, **kwargs):
    if name == "optim":
        return get_optim_dataset(
            path=os.path.join(BASE_PATH, "evaluation-data", "optim-data"), 
            n_channels=kwargs.pop("n_channels", 1),
            transform=kwargs.pop("transform", None),
            training=kwargs.pop("training", False),
            batch_size=kwargs.pop("batch_size", 1),
            num_samples=kwargs.pop("num_samples", None),
            **kwargs
            )
    elif name == "synaptic-proteins":
        return get_synaptic_proteins_dataset(
            path=os.path.join(BASE_PATH, "evaluation-data", "TheresaProteins"), 
            n_channels=kwargs.get("n_channels", 1), 
            transform=kwargs.get("transform", None),
            batch_size=kwargs.get("batch_size", 1),
            num_samples=kwargs.get("num_samples", None),
            )
    
    elif name == "neural-activity-states":
        return get_neural_activity_states(
            path=os.path.join(BASE_PATH, "evaluation-data", "NeuralActivityStates"),
            n_channels=kwargs.get("n_channels", 1), 
            transform=kwargs.get("transform", None),
            batch_size=kwargs.get("batch_size", 1),
            num_samples=kwargs.get("num_samples", None),
            balance=kwargs.get("balance", True),
            protein_id=3
        )
    elif name == "factin-rings-fibers":
        return get_factin_rings_fibers_dataset(path=path, transform=kwargs.get("transform", 1))
    elif name == "factin-block-glugly":
        return get_factin_block_glugly_dataset(path=path, transform=kwargs.get("transform", 1))
    elif name == "peroxisome":
        return get_peroxisome_dataset(
            path=os.path.join(BASE_PATH, "evaluation-data", "peroxisome"), 
            superclasses=kwargs.get("superclasses", False),
            n_channels=kwargs.get("n_channels", 1), 
            transform=kwargs.get("transform", None),
            batch_size=kwargs.get("batch_size", 64),
            num_samples=kwargs.get("num_samples", None),
            balance=kwargs.get("balance", True),
        )
    elif name == "polymer-rings":
        return get_polymer_rings_dataset(
            path=os.path.join(BASE_PATH, "evaluation-data", "polymer-rings"), 
            superclasses=kwargs.get("superclasses", True),
            n_channels=kwargs.get("n_channels", 1), 
            transform=kwargs.get("transform", None),
            batch_size=kwargs.get("batch_size", 64),
            num_samples=kwargs.get("num_samples", None),
        )
    elif name == "dl-sim":
        return get_dl_sim_dataset(
            path=os.path.join(BASE_PATH, "evaluation-data", "DL-SIM"), 
            n_channels=kwargs.get("n_channels", 1), 
            transform=kwargs.get("transform", None),
            batch_size=kwargs.get("batch_size", 64),
            num_samples=kwargs.get("num_samples", None),
        )    
    elif name == "resolution":
        return get_resolution_dataset(
            path=os.path.join(BASE_PATH, "evaluation-data", "resolution-dataset"), 
            n_channels=kwargs.get("n_channels", 1), 
            transform=kwargs.get("transform", None),
            batch_size=kwargs.get("batch_size", 64),
            num_samples=kwargs.get("num_samples", None),
        )
    else:
        raise NotImplementedError(f"`{name}` dataset is not supported.")




    

