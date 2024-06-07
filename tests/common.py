import numpy as np;

import torch
import torch.nn as nn
import torch.nn.functional as F

def pretty_logits(logits, max_ = 0xFFFFFFFF, decimals=4):
    logits0 = logits.contiguous().view(-1)
    row = "["
    cc = len(logits0)
    n = min(cc, max_)
    for j in range(n):
        val = round(logits0[j].item(), decimals)
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

