using System;
using System.IO;

using nn;

unsafe internal static class generic {
    // static void reset_weights(Linear layer, IRNG g, string nonlinearity = "leaky_relu") {
    //     nn.init.kaiming_uniform_(
    //         layer._Weight.data,
    //         layer._Weight.numel(),
    //         g,
    //         layer.I,
    //         (float)Math.Sqrt(5),
    //         nonlinearity);
    // 
    //     if (layer._Bias != null) {
    //         nn.rand.uniform_(
    //             layer._Bias.data,
    //             layer._Bias.numel(),
    //             g,
    //             -(float)(1.0 / Math.Sqrt(layer.I)),
    //             (float)(1.0 / Math.Sqrt(layer.I)));
    //     }
    // }

    public static void eval_net(TextWriter Console, IModel net, Linear W, Tensor input, bool grad) {
        net.eval();
        var logits = net.forward(input);
        Console.WriteLine("logits:");
        Console.WriteLine(Common.pretty_logits(logits.data, logits.numel(), 0xFFFFFFFF));
        Console.WriteLine("weight:");
        Console.WriteLine(Common.pretty_logits(W._Weight.data, W._Weight.numel(), 0xFFFFFFFF));
        if (W._Weight.grad != null && grad) {
            Console.WriteLine("weight.grad:");
            Console.WriteLine(Common.pretty_logits(W._Weight.grad, W._Weight.numel(), 0xFFFFFFFF));
        }
        if (W._Bias != null) {
            Console.WriteLine("bias:");
            Console.WriteLine(Common.pretty_logits(W._Bias.data, W._Bias.numel(), 0xFFFFFFFF));
            if (W._Bias.grad != null && grad) {
                Console.WriteLine("bias.grad:");
                Console.WriteLine(Common.pretty_logits(W._Bias.grad, W._Bias.numel(), 0xFFFFFFFF));
            }
        }
    }

    public static void test_net(TextWriter Console,
        bool bias,
        string activation,
        string optim,
        string loss,
        float lr,
        float momentum,
        float weight_decay,
        int decimals) {

        const int I = 5;
        const int H = 7;
        const int O = 3;
        const int B = 4;

        Console.WriteLine($"<test_net>");

        var g = new nn.rand.mt19937(137);
        Linear W_hidden = new nn.Linear(I, H, bias: bias, Linear.Kernel.Naive);
        Linear W_output = new nn.Linear(H, O, bias: bias, Linear.Kernel.Naive);
        nn.init.reset_weights(W_hidden, "leaky_relu", g);
        nn.init.reset_weights(W_output, "leaky_relu", g);
        IModel F_act = new nn.Identity();
        switch (activation) {
            case "Identity":
                break;
            case "Tanh":
                F_act = new nn.Tanh();
                break;
            case "LeakyReLU":
                F_act = new nn.LeakyReLU();
                break;
            case "Dropout":
                F_act = new nn.Dropout(g);
                break;
            case "ReLU":
                F_act = new nn.ReLU();
                break;
            case "Sigmoid":
                F_act = new nn.Sigmoid();
                break;
            default:
                if (!string.IsNullOrEmpty(activation)) {
                    throw new ArgumentOutOfRangeException($"The specified activation '{activation}' is not supported");
                }
                break;
        }

        var net = new nn.Sequential(
            W_hidden,
            F_act,
            W_output,
            loss == "BCELoss" ? new Sigmoid() : new Identity());

        var input = Tensor.ones((uint)I * B, requires_grad: true);
        var target = Tensor.ones((uint)O * B, requires_grad: false);
        eval_net(Console, net, W_output, input, grad: false);

        IOptimizer optimizer = null;
        if (optim == "SGD") {
            optimizer = new nn.SGD(net.parameters(), lr: lr, momentum: momentum, weight_decay: weight_decay);
        } else if (optim == "AdamW") {
            optimizer = new nn.AdamW(net.parameters(), lr: lr, weight_decay: weight_decay);
        } else {
            throw new ArgumentOutOfRangeException($"The specified optimizer '{optim}' is not supported");
        }
        for (int step = 0; step < 1000; step++) {
            net.train();
            var logits = net.forward(input);
            var diff = 0.0;
            if (loss == "MSELoss") {
                diff = F.mse_loss(
                    logits.data,
                    logits.grad,
                    logits.numel(),
                    target.data);
            } else if (loss == "BCELoss") {
                diff = F.binary_cross_entropy(
                    logits,
                    target);
            } else {
                throw new ArgumentOutOfRangeException($"The specified loss '{loss}' is not supported");
            }
            optimizer.zero_grad();
            net.backward(logits);
            optimizer.step();
            if (step % 100 == 0) {
                Console.WriteLine($"[{step}]: [{Math.Round(diff, decimals):f4}]");
            }
        }

        eval_net(Console, net, W_output, input, grad: true);
        Console.WriteLine("</test_net>");
    }
}