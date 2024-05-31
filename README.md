# nn.cs

### Neural network training in simple raw C#

The base reference implementation follows [PyTorch](https://github.com/pytorch/pytorch), in both the naming convention and style, as much as possible with the goal of achieving numerically identical results in **C#**

e.g.:

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

While [PyTorch](https://github.com/pytorch/pytorch) and many other frameworks build a compute graph for automatic gradients, it is not the goal of **nn.cs** at this time.


In **nn.cs** you implement the forward and backward kernels by hand or choose from a collection of built-in ones.
