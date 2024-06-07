using System;
using System.IO;

using nn;

unsafe internal static class dropout {
    public static void test_dropout_forward(TextWriter Console, double p, bool train) {
        Console.WriteLine($"<test_dropout_forward p={p:f1} mode='{(train ? "train" : "eval")}'>");
        var g = new nn.rand.mt19937(137);
        var a = Tensor.NaN(17);
        Console.WriteLine(g.randint32());
        nn.rand.normal_(a.data, a.numel(), g, 0, 1);
        Console.WriteLine(Common.pretty_logits(a.data, a.numel()));
        Console.WriteLine(g.randint32());
        var mask = Tensor.NaN(7);
        F.dropout_forward_cpu(
            a.data,
            a.data,
            mask.data,
            a.numel(),
            p,
            train,
            g);
        Console.WriteLine(Common.pretty_logits(a.data, a.numel(), 0xFFFFFFFF));
        Console.WriteLine(g.randint32());
        Console.WriteLine("</test_dropout_forward>");
    }

    public static void test_dropout_backward(TextWriter Console, double p, bool use_dropout) {
        Console.WriteLine($"<test_dropout_backward p={p:f1} dropout='{(use_dropout ? "yes" : "no")}'>");
        var g = new nn.rand.mt19937(137);
        Console.WriteLine(g.randint32());
        Linear hidden = new nn.Linear(7, 7, bias: false);
        nn.init.kaiming_uniform_(
            hidden._Weight.data,
            hidden._Weight.numel(),
            g,
            hidden.I,
            (float)Math.Sqrt(5));
        Dropout dropout = use_dropout ? new Dropout(g, p) : null;
        Tanh tanh = new nn.Tanh();
        Console.WriteLine(g.randint32());
        Console.WriteLine("weight:");
        Console.WriteLine(Common.pretty_logits(hidden._Weight.data, hidden._Weight.numel(), 0xFFFFFFFF));
        var input = Tensor.zeros(7, requires_grad: true);
        nn.rand.normal_(input.data, input.numel(), g, 0, 1);
        Console.WriteLine(g.randint32());
        var output = hidden.forward(input);
        if (dropout != null) {
            output = dropout.forward(output);
        }
        output = tanh.forward(output);
        Console.WriteLine("output:");
        Console.WriteLine(Common.pretty_logits(output.data, output.numel(), 0xFFFFFFFF));
        Console.WriteLine(g.randint32());
        var target = Tensor.ones(7);
        var loss = F.mse_loss(
            output.data,
            output.grad,
            output.numel(),
            target.data);
        Console.WriteLine("loss:");
        Console.WriteLine($"[{loss:f4}]");
        output = tanh.backward(output);
        if (dropout != null) {
            output = dropout.backward(output);
        }
        hidden.backward(output);
        Console.WriteLine(g.randint32());
        Console.WriteLine("weight.grad:");
        Console.WriteLine(Common.pretty_logits(hidden._Weight.grad, hidden._Weight.numel(), 0xFFFFFFFF));
        Console.WriteLine(g.randint32());
        Console.WriteLine("</test_dropout_backward>");
    }
}