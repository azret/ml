# Machine Learning in C#

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build](https://github.com/azret/ml/actions/workflows/ci.yml/badge.svg)](https://github.com/azret/nn.cs/actions/workflows/ci.yml)

### Quick Start

```csharp
nn.Sequential model = new nn.Sequential(
    new nn.Linear(16, 4 * 16, bias: true),
    new nn.ReLU(),
    new nn.Dropout(_RNG_, 0.1),
    new nn.Linear(4 * 16, 16, bias: true),
    new nn.Sigmoid()
);

var optimizer = new nn.AdamW(model.parameters(), lr=1e-4f);

for (uint iter = 0; iter < 100; iter++) {
    var logits = model.forward(_INPUT_);

    var loss = F.binary_cross_entropy(
        logits,
        _TARGET_);

    optimizer.zero_grad();

    model.backward(logits);

    optimizer.step();
}
```
