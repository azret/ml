# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

import numpy as np;

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import pretty_logits

def eval_net(net, W, input):
    net.eval()
    logits = net(input)
    print(f"logits:")
    print(pretty_logits(logits, 0xFFFFFFFF))
    print("weight:")
    print(pretty_logits(W.weight, 0xFFFFFFFF))
    if W.weight.grad is not None:
        print("weight.grad:")
        print(pretty_logits(W.weight.grad, 0xFFFFFFFF))
    if W.bias is not None:
        print("bias:")
        print(pretty_logits(W.bias, 0xFFFFFFFF))
        if W.bias.grad is not None:
            print("bias.grad:")
            print(pretty_logits(W.bias.grad, 0xFFFFFFFF))

def test_net(optim, lr, bias, activation, momentum, weight_decay, decimals):
    I = 5
    H = 7
    O = 3
    B = 4

    print("<test_net>")

    torch.manual_seed(137)
    W_hidden = nn.Linear(I, H, bias=bias)
    W_output = nn.Linear(H, O, bias=bias)

    if activation == "Identity":
        F_act = nn.Identity()
    elif activation == "ReLU":
        F_act = nn.ReLU()
    elif activation == "Sigmoid":
        F_act = nn.Sigmoid()
    elif activation == "LeakyReLU":
        F_act = nn.LeakyReLU()
    elif activation == "Dropout":
        F_act = nn.Dropout()
    elif activation == "Tanh":
        F_act = nn.Tanh()

    net = nn.Sequential(W_hidden, F_act, W_output);

    sample = torch.ones(B, I)
    target = torch.ones(B, O)

    eval_net(net, W_output, sample)

    if optim == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(net.parameters(), lr = lr, weight_decay=weight_decay)

    for step in range(1000):
        net.train()
        logits = net(sample)
        loss = F.mse_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"[{step}]: {pretty_logits(loss, decimals=decimals)}")

    eval_net(net, W_output, sample)
    print("</test_net>")

if __name__ == "__main__":
    import argparse

    from pathlib import Path

    path = Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser()
    parser.add_argument("--bias", type=str, default="yes")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--maxDegreeOfParallelism", type=int, default=-1)
    parser.add_argument("--activation", type=str, default="Identity")
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--decimals", type=int, default=4)

    args = parser.parse_args()

    assert args.activation in { "Identity", "ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Dropout" }
    assert args.optim in { "SGD", "AdamW" }
    assert args.bias in { "yes", "no" }
    assert 0 <= args.lr <= 1
    assert 0 <= args.momentum <= 1
    assert 0 <= args.weight_decay <= 1

    test_net(optim = args.optim,
             lr = args.lr,
             bias = True if args.bias == "yes" else False,
             activation = args.activation,
             momentum = args.momentum,
             weight_decay = args.weight_decay,
             decimals=args.decimals);
