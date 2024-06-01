# Neural network training in C#

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build](https://github.com/azret/nn.cs/actions/workflows/ci.yml/badge.svg)](https://github.com/azret/nn.cs/actions/workflows/ci.yml)

### Quick Start

```csharp
nn.Sequential model = new nn.Sequential(
    new nn.Linear(16, 4 * 16, bias: true),
    new nn.Sigmoid(),
    new nn.Linear(4 * 16, 16, bias: true),
    new nn.Sigmoid()
);

var optimizer = new nn.AdamW(model.parameters());

for (uint iter = 0; iter < 100; iter++) {
    var logits = model.forward(input);

    optimizer.zero_grad();

    var loss = F.binary_cross_entropy(
        logits,
        target);

    model.backward(logits);

    optimizer.step();
}
```
