import torchvision




class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 0.10
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        self.train_transform = torchvision.transforms.Compose(
            [   
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )

        ## Use this series of transformations if SimCLR pretrained on optim data
        # self.train_transform = torchvision.transforms.Compose(
        #     [   
        #         torchvision.transforms.RandomResizedCrop(size=size, scale=(0.375, 1.0)),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.Normalize([0.5001, 0.5001, 0.5001], [0.0003, 0.0003, 0.0003])

        #     ]
        # )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)     
