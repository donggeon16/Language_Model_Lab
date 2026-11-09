
import random
import numpy as np
import pytest
import torch

from lm_lab.utils.seed import seed_everything


def test_seed_reproducibility():
    seed_everything(42)

    python_a = random.random()
    numpy_a = np.random.rand(3, 3)
    torch_a = torch.rand(3, 3)

    seed_everything(42)

    python_b = random.random()
    numpy_b = np.random.rand(3, 3)
    torch_b = torch.rand(3, 3)

    assert python_a == python_b
    assert np.array_equal(numpy_a, numpy_b)
    assert torch.equal(torch_a, torch_b)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_cuda_seed_reproducibility():
    seed_everything(42)
    a = torch.rand(3, 3, device="cuda")

    seed_everything(42)
    b = torch.rand(3, 3, device="cuda")

    assert torch.equal(a, b)
