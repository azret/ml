# Neural network training in C#

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build](https://github.com/azret/nn.cs/actions/workflows/ci.yml/badge.svg)](https://github.com/azret/nn.cs/actions/workflows/ci.yml)

### Quick Start

```csharp
// Use vectorized MatMul kernel (AVX2)
using Linear = nn.Linear<nn.CPU.MatMulAVX2>;

nn.Sequential model = new nn.Sequential(
    new Linear(16, 4 * 16, bias: true),
    new nn.ReLU(),
    new Linear(4 * 16, 16, bias: true),
    new nn.Sigmoid()
);

var optimizer = new nn.AdamW(model.parameters(), lr=1e-4f);

for (uint iter = 0; iter < 100; iter++) {
    var logits = model.forward(input);

    var loss = F.binary_cross_entropy(
        logits,
        target);

    optimizer.zero_grad();

    model.backward(logits);

    optimizer.step();
}
```
