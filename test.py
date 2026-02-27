
from stedfm.datasets import get_dataset

if __name__ == "__main__":

    dataset = get_dataset(
        name="factin",
        path=None,
        transform=None,
    )

    print(f"Dataset size: {len(dataset)} samples")