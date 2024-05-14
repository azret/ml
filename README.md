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

e.g. A naive **C#** implementation (which is pretty slow) can be re-written in plain **C**, compiled with the latest state of the art **C**/**C++** compiler (with full optimization, fast math, auto-vectorization etc...) and then imported into **nn.cs** as a native CPU kernel giving a ~10x to ~20x performance boost.

```csharp
public static unsafe void sigmoid_forward_cpu(
    float* _Out,       /* [N] */
    float* _In,        /* [N] */
    uint N) {

    for (int n = 0; n < N; n++) {
        var σ = 1.0f / (1.0f + (float)Math.Exp(-_In[n]));
        _Out[n] = σ;
    }
}

public static unsafe void sigmoid_backward_cpu(
    Tensor _Out,       /* [N] */
    Tensor _In,        /* [N] */
    uint N) {

    for (int n = 0; n < N; n++) {
        var σ = 1.0f / (1.0f + (float)Math.Exp(-_In.data[n]));
        _In.grad[n] += σ * (1.0f - σ) * _Out.grad[n];
    }
}
```

