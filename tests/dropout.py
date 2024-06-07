import numpy as np;

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import pretty_logits

def test_dropout_forward(p, train):
    print("<test_dropout_forward p=" + str(p) + " mode='" + ("train" if train else "eval") + "'>")
    torch.manual_seed(137)
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    model = nn.Dropout(p)
    if train:
        model.train();
    else:
        model.eval();
    input = torch.randn(17)
    print(pretty_logits(input))
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    output = model(input);
    print(pretty_logits(output, 0xFFFFFFFF))
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print("</test_dropout_forward>")

class Net(nn.Module):
    def __init__(self, input_size, num_classes, dropout, p):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_size, num_classes, bias=False)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(p)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.hidden(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.tanh(x)
        return x

def test_dropout_backward(use_dropout, p):
    print("<test_dropout_backward p=" + str(p) + " dropout='" + ("yes" if use_dropout else "no") + "'>")
    torch.manual_seed(137)
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    model = Net(7, 7, use_dropout, p)
    model.train()
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print("weight:")
    print(pretty_logits(model.hidden.weight, 0xFFFFFFFF))
    input = torch.randn(7)
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    output = model(input)
    print("output:")
    print(pretty_logits(output, 0xFFFFFFFF))
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    target = torch.ones(7)
    loss = F.mse_loss(output, target)
    print("loss:")
    print(pretty_logits(loss))
    loss.backward()
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print("weight.grad:")
    print(pretty_logits(model.hidden.weight.grad, 0xFFFFFFFF))
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print("</test_dropout_backward>")

if __name__ == "__main__":
    import argparse

    from pathlib import Path

    path = Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--net", type=str, default="no")
    parser.add_argument("--dropout", type=str, default="yes")
    args = parser.parse_args()
    
    assert args.mode in {"train", "eval"}
    assert 0 <= args.p <= 1
    assert args.net in {"yes", "no"}
    assert args.dropout in {"yes", "no"}
    
    if args.net == "no":
        if args.mode == "train":
            test_dropout_forward(args.p, True);
        elif args.mode == "eval":
            test_dropout_forward(args.p, False);
    elif args.net == "yes":
        if args.dropout == "yes":
            test_dropout_backward(use_dropout = True, p = args.p);
        elif args.dropout == "no":
            test_dropout_backward(use_dropout = False, p = args.p);
