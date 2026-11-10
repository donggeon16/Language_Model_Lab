
import pytest
import torch
import torch.nn.functional as F

from lm_lab.training.losses import stable_cross_entropy


@pytest.mark.parametrize(
    "logits_data, labels_data",
    [
        # Case 1: 균등확률
        ([[0.0, 0.0, 0.0]], [1]),

        # Case 2: 다중 Batch
        ([[2.0, 1.0, 0.0],
          [0.0, 3.0, 1.0]], [0, 1]),

        # Case 3: 극단적인 Logits
        ([[1000.0, 1001.0, 999.0]], [1]),
    ],
)
def test_cross_entropy_values(logits_data, labels_data):
    logits = torch.tensor(logits_data, dtype=torch.float64)
    labels = torch.tensor(labels_data, dtype=torch.long)

    custom_loss = stable_cross_entropy(logits, labels)
    reference_loss = F.cross_entropy(logits, labels)

    # 1. Loss Shape 확인 (Scalar)
    assert custom_loss.ndim == 0

    # 2. NaN / Inf 검사
    assert torch.isfinite(custom_loss).item()

    # 3. 공식 구현과 Loss 값 비교
    torch.testing.assert_close(
        custom_loss,
        reference_loss,
        atol=1e-10,
        rtol=1e-10,
    )
