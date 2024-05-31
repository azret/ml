using System;

using nn;
using static nn.F;

internal unsafe class regression {

    static string pretty_logits(float[] logits) {
        fixed (float* logits0 = logits) {
            return pretty_logits(logits0, (uint)logits.Length);
        }
    }
    static string pretty_logits(float* logits0, uint cc) {
        string row = "[";
        for (int j = 0; j < Math.Min(cc, 7); j++) {
            row += $"{(Math.Round(logits0[j], 4))}";
            if (j == cc - 1)
                row += "";
            else if (j > 0)
                row += ", ";
            else
                row += ", ";
        }
        row += "]";
        return row;
    }

    static void Main() {
        var g = new nn.rand.mt19937(137);

        nn.Linear<MatMulAVX2> L1, L2;

        nn.Sequential model = new nn.Sequential(
            L1 = new nn.Linear<MatMulAVX2>(768, 4 * 768, bias: true),
            new nn.Sigmoid(),
            L2 = new nn.Linear<MatMulAVX2>(4 * 768, 768, bias: true),
            new nn.Sigmoid()
        );

        nn.rand.kaiming_uniform_(
            L1._Weight.data,
            L1._Weight.numel(),
            g,
            checked((int)L1._Weight.numel()),
            (float)Math.Sqrt(5));

        nn.rand.uniform_(
            L1._Bias.data,
            L1._Bias.numel(),
            g,
            -(float)(1.0 / Math.Sqrt(checked((int)L1._Weight.numel()))),
            (float)(1.0 / Math.Sqrt(checked((int)L1._Weight.numel()))));

        nn.rand.kaiming_uniform_(
            L2._Weight.data,
            L2._Weight.numel(),
            g,
            checked((int)L1._Weight.numel()),
            (float)Math.Sqrt(5));

        nn.rand.uniform_(
            L2._Bias.data,
            L2._Bias.numel(),
            g,
            -(float)(1.0 / Math.Sqrt(checked((int)L1._Weight.numel()))),
            (float)(1.0 / Math.Sqrt(checked((int)L1._Weight.numel()))));

        const int batch_size = 1024;

        var x = Tensor.zeros(batch_size * 768);
        var y = new float[batch_size * 768];
        for (int i = 0; i < y.Length; i++) {
            y[i] = 1f;
        }

        var lr = 1e-3f;

        var optimizer = new nn.AdamW(model.parameters(), lr: 0);

        Console.WriteLine($"parameters: {optimizer.get_num_params()}");

        int epochs = 100;

        for (uint epoch = 0; epoch < epochs; epoch++) {
            var start_time = kernel32.millis();
            var logits = model.forward(x);
            optimizer.zero_grad();

            // var loss = F.binary_cross_entropy(logits, y);
            var loss = F.no_loss(logits, y);

            model.backward(logits);
            optimizer.step();

            var elapsedMillis = (kernel32.millis() - start_time);

            Console.WriteLine($"{epoch}: lr = {lr}, loss={loss:f6}, {elapsedMillis} ms");

            if (epoch == epochs - 1)
                Console.WriteLine(pretty_logits(logits.data, logits.numel()));
        }

        Console.ReadKey();
    }
}