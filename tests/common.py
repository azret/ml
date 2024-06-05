import numpy as np;

import torch
import torch.nn as nn
import torch.nn.functional as F

def pretty_logits(logits, max_ = 7):
    logits0 = logits.view(-1)
    row = "["
    cc = len(logits0)
    n = min(cc, max_)
    for j in range(n):
        val = logits0[j].item()
        if (val == -0):
            val = 0;
        row += f"{val:.4f}"
        if j == n - 1:
            if (n < cc):
                row += ", ..."
        else:
            row += ", "
    row += "]"
    return row

