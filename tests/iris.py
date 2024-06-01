import numpy as np;

import torch
import torch.nn as nn
import torch.nn.functional as F

# def kaiming_uniform_(
#     tensor: torch.Tensor,
#     a: float = 0,
#     mode: str = "fan_in",
#     nonlinearity: str = "leaky_relu"):
#     if 0 in tensor.shape:
#         torch.warnings.warn("Initializing zero-element tensors is a no-op")
#         return tensor
#     fan = torch.nn.init._calculate_correct_fan(tensor, mode)
#     gain = torch.nn.init.calculate_gain(nonlinearity, a)
#     std = gain / torch.math.sqrt(fan)
#     bound = torch.math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
#     with torch.no_grad():
#         return tensor.uniform_(-bound, bound)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.id1 = nn.Identity()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)
        self.id2 = nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.id1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.id2(out)
        out = self.sigmoid(out)
        return out

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    # def reset_parameters(self, linear) -> None:
    #     torch.manual_seed(137)
    #     print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    #     kaiming_uniform_(linear.weight, a=torch.math.sqrt(5))
    #     print(f"weight: {pretty_logits(linear.weight)}")
    #     print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    #     if linear.bias is not None:
    #         fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(linear.weight)
    #         bound = 1 / torch.math.sqrt(fan_in) if fan_in > 0 else 0
    #         print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    #         torch.nn.init.uniform_(linear.bias, -bound, bound)
    #         print(f"bias: {pretty_logits(linear.bias)}")
    #         print(torch.randint(0, 0xFFFFFFFF, [1]).item())

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

def pretty_array(logits0):
    row = "["
    for j in range(min(len(logits0), 7)):
        row += f"{(round(logits0[j], 4)):.4f}"
        if j == len(logits0) - 1:
            row += ""
        else:
            row += ", "
    row += "]"
    return row

if __name__ == "__main__":
    import argparse

    from pathlib import Path

    path = Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="MSELoss")
    args = parser.parse_args()

    assert 1 <= args.batch_size <= 1024
    assert args.optim in {"SGD", "AdamW"}
    assert 0 <= args.lr <= 1
    assert args.loss in {"BCELoss", "MSELoss"}

    data = np.loadtxt(str(path) + '//iris.csv', usecols=range(0,7), delimiter=",", skiprows=0, dtype=np.float32)

    def get_batch(B):
        assert B <= len(data), "not enough items for the specified batch size"
        i = 0
        while True:
            take = B
            if (i + take > len(data)):
                take = len(data) - i
            x = (data[i:i+take][:,[0, 1, 2, 3]])
            y = (data[i:i+take][:,[4, 5, 6]])
            yield x, y, take
            i += take
            if i >= len(data):
                i = 0

    # test data loader

    # data_iter_test = iter(get_batch(37))
    # for i in range(5):
    #     x, y, take = next(data_iter_test)
    #     print(f"batch_size: {take}:")
    #     for k in range(len(x)):
    #         print(f"x: {pretty_array(x[k])} = y: {pretty_array(y[k])}")

    torch.manual_seed(137)

    model = Net(input_size=4, hidden_size=8, num_classes=3)

    # test weight initialization

    print(f"fc1.weight: {pretty_logits(model.fc1.weight)}")
    if model.fc1.bias is not None:
        print(f"fc1.bias: {pretty_logits(model.fc1.bias)}")
    print(f"fc2.weight: {pretty_logits(model.fc2.weight)}")
    if model.fc2.bias is not None:
        print(f"fc2.bias: {pretty_logits(model.fc2.bias)}")

    if args.loss == "BCELoss":
        criterion = nn.BCELoss()
    if args.loss == "MSELoss":
        criterion = nn.MSELoss()

    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"parameters: {model.get_num_params()}")

    num_epochs = 10

    data_iter = iter(get_batch(args.batch_size))

    for epoch in range(num_epochs):
        x, y, take = next(data_iter)
        logits = model(torch.from_numpy(x))
        print(f'{epoch}: logits: {pretty_logits(logits)}')
        loss = criterion(logits, torch.from_numpy(y))
        print(f'{epoch}: loss: {loss.item():.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"{epoch}: fc2.weight.grad: {pretty_logits(model.fc2.weight.grad)}")
        if model.fc2.bias is not None:
            print(f"{epoch}: fc2.bias.grad: {pretty_logits(model.fc2.bias.grad)}")
        print(f"{epoch}: fc1.weight.grad: {pretty_logits(model.fc1.weight.grad)}")
        if model.fc1.bias is not None:
            print(f"{epoch}: fc1.bias.grad: {pretty_logits(model.fc1.bias.grad)}")
        print(f"{epoch}: fc1.weight: {pretty_logits(model.fc1.weight)}")
        if model.fc1.bias is not None:
            print(f"{epoch}: fc1.bias: {pretty_logits(model.fc1.bias)}")
        print(f"{epoch}: fc2.weight: {pretty_logits(model.fc2.weight)}")
        if model.fc2.bias is not None:
            print(f"{epoch}: fc2.bias: {pretty_logits(model.fc2.bias)}")

