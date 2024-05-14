# nn.cs

### Neural network training in simple raw C#

The base reference implementation follows [PyTorch](https://github.com/pytorch/pytorch) as much as possible, in both the naming convention and style, with the goal of achieving numerically identical results in **C#**

e.g.

```csharp
nn.Sequential model = new nn.Sequential(
    new nn.Linear(16, 1, bias: true),
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
