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
    assert abs(ag.item() - fdg.item()) < 1e-5 # type: ignore
    assert abs(ag.item() - 42.0) < 1e-5 # type: ignore

def matrix_loss(x, w):
    y = x @ w
    return (y * y).sum()

def matrix_finite_difference(f, x, w, h=1e-5):
    x_plus = x.detach().clone()
    x_plus[0, 0] = x_plus[0, 0] + h
    x_minus = x.detach().clone()
    x_minus[0, 0] = x_minus[0, 0] - h
    loss_plus = f(x_plus, w.detach())
    loss_minus = f(x_minus, w.detach())
    return (loss_plus - loss_minus) / (2 * h)

def test_matrix_autograd():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.float64, requires_grad=True)
    loss = matrix_loss(x, w)
    loss.backward()
    xg = x.grad
    wg = w.grad
    fdg = matrix_finite_difference(matrix_loss, x, w)
    assert xg.shape == (2, 3)
    assert wg.shape == (3, 4)
    assert abs(xg[0, 0].item() - fdg.item()) < 1e-5
    assert abs(xg[0, 0].item() - 112.0) < 1e-5