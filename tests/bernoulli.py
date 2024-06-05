import numpy as np;

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import pretty_logits

if __name__ == "__main__":
    print("<bernoulli_>")
    torch.manual_seed(137)
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    a = torch.zeros(137)
    a.uniform_(0, 1)
    print(pretty_logits(a))
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    a.bernoulli_(p = 0.2);
    print(pretty_logits(a, 137))
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print("</bernoulli_>")
