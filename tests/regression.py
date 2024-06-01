# See https://github.com/pytorch/examples/blob/main/regression/main.py

import torch
import torch.nn.functional as F

def pretty_logits(logits):
    logits0 = logits.view(-1)
    row = "["
    for j in range(min(len(logits0), 7)):
        row += f"{(round(logits0[j].item(), 4)):.4f}"
        if j == len(logits0) - 1:
            row += ""
        elif j > 0 and ((j + 1) % logits.shape[0]) == 0:
            row += ", "
        else:
            row += ", "
            
    row += "]"
    return row

torch.manual_seed(137)

print(torch.randint(0, 0xffffffff, [1]).item())

POLY_DEGREE = 4

W_target = torch.zeros(POLY_DEGREE, 1)
b_target = torch.zeros(1)

W_target.uniform_(0, 1)
b_target.uniform_(0, 1)

print(torch.randint(0, 0xffffffff, [1]).item())

print("W_target: ", pretty_logits(W_target))
print("b_target: ", pretty_logits(b_target))

model = torch.nn.Linear(W_target.size(0), 1)

print("fc.weight: ", pretty_logits(model.weight))
print("fc.bias: ", pretty_logits(model.bias))

print(torch.randint(0, 0xffffffff, [1]).item())

def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target.item()

def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y


for step in range(0, 10000):
    batch_x, batch_y = get_batch()
    
    if (step % 1000 == 0):
        print("batch_x: ", pretty_logits(batch_x))

    model.zero_grad()

    output = F.mse_loss(model(batch_x), batch_y)
    loss = output.item()

    output.backward()

    for param in model.parameters():
        param.data.add_(-0.001 * param.grad)

    if (step % 1000 == 0):
        print(f"step # {step}, loss = {loss}")

print('Loss: {:.6f} after {} steps'.format(loss, step))
print('==> Learned function:\t' + poly_desc(model.weight.view(-1), model.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))