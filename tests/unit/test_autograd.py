import torch

def finite_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return (3 * x + 1) ** 2

def test_scalar_autograd():
    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    y = f(x)
    y.backward()
    ag = x.grad
    fdg = finite_difference(f, x.detach())
    assert abs(ag.item() - fdg.item()) < 1e-5
    assert abs(ag.item() - 42.0) < 1e-5