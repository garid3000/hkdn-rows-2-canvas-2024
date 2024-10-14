import os
import numpy as np
from numpy.typing import NDArray

def get_sampling_xy(path2npy: str) -> NDArray:
    x = np.load(path2npy)["angl"]
    result = np.empty(shape=(x.shape[0], 2))
    result[:, 0] = int(os.path.basename(path2npy).split("_")[-1].split(".")[0])
    result[:, 1] = x[:, 1]
    return result
