import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Python, NumPy, PyTorch의 난수 생성기를 초기화"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
