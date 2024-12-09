def denormalize(img: np.ndarray) -> np.ndarray:
    return (img + 1.0) / 2.0