import torch


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    m = torch.max(logits, dim=1, keepdim=True).values
    stable_logits = logits - m

    exp_sum = torch.exp(stable_logits).sum(dim=1, keepdim=True)
    log_exp_sum = torch.log(exp_sum)

    return stable_logits - log_exp_sum


def stable_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
) -> torch.Tensor:
    log_probs = log_softmax(logits)
    batch_indices = torch.arange(
        logits.shape[0],
        device=logits.device,
    )
    correct_log_probs = log_probs[batch_indices, labels]
    loss = -correct_log_probs.mean()

    return loss


def main():
    logits = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    labels = torch.tensor([1])
    loss = stable_cross_entropy(logits=logits, labels=labels)
    print("Loss:", loss)


if __name__ == "__main__":
    main()