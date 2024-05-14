using System;

using nn;

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

    const int POLY_DEGREE = 4;

    static string poly_desc(Tensor W, Tensor b) {
        string result = "y = ";
        for (int i = 0; i < W.numel(); i++) {
            result += $"+{W.data[i]:f2} x^{i + 1} ";
        }
        result += $"+{b.data[0]:f2}";
        return result;
    }

    static nn.Tensor W_target = new nn.Tensor(POLY_DEGREE, requires_grad: false);
    static nn.Tensor b_target = new nn.Tensor(1, requires_grad: false);

    static (float[], float[]) get_batch(nn.rand.IRNG g, int batch_size = 32) {
        float[] random = new float[batch_size];
        nn.rand.normal_(random, g);
        var x = new float[batch_size * POLY_DEGREE];
        for (int b = 0; b < batch_size; b++) {
            for (int p = 0; p < POLY_DEGREE; p++) {
                x[b * POLY_DEGREE + p] = (float)Math.Pow(random[b], p + 1);
            }
        }
        var y = new float[batch_size];
        for (int b = 0; b < batch_size; b++) {
            y[b] = 0;
            for (int p = 0; p < POLY_DEGREE; p++) {
                y[b] += W_target.data[p] * x[b * POLY_DEGREE + p];
            }
            y[b] += b_target.data[0];
        }
        return (x, y);
    }

    static void Main() {
        var g = new nn.rand.mt19937(137);

        Console.WriteLine(g.randint32());

        nn.rand.uniform_(W_target.data, W_target.numel(), g);
        nn.rand.uniform_(b_target.data, b_target.numel(), g);

        Console.WriteLine(g.randint32());

        Console.WriteLine($"W_target: {pretty_logits(W_target.data, W_target.numel())}");
        Console.WriteLine($"b_target: {pretty_logits(b_target.data, b_target.numel())}");

        var fc = new nn.Linear<F.MatMulV>(POLY_DEGREE, 1);

        nn.rand.kaiming_uniform_(
            fc._Weight.data,
            fc._Weight.numel(),
            g,
            POLY_DEGREE,
            (float)Math.Sqrt(5));

        nn.rand.uniform_(
            fc._Bias.data,
            fc._Bias.numel(),
            g,
            -(float)(1.0/Math.Sqrt(POLY_DEGREE)),
            (float)(1.0 / Math.Sqrt(POLY_DEGREE)));

        Console.WriteLine($"fc.weight: {pretty_logits(fc._Weight.data, fc._Weight.numel())}");
        Console.WriteLine($"fc.bias: {pretty_logits(fc._Bias.data, fc._Bias.numel())}");

        Console.WriteLine(g.randint32());

        double loss = 0; int step;

        for (step = 0; step < 10000; step++) {
            var batch = get_batch(g);

            if (step % 1000 == 0)
                Console.WriteLine($"batch_x: {pretty_logits(batch.Item1)}");

            fc._Weight.zero_grad();
            fc._Bias.zero_grad();

            var input = new Tensor(batch.Item1);

            var logits = fc.forward(input);

            input.Dispose();

            loss = F.mse_loss(logits, batch.Item2);

            fc.backward(logits);

            foreach (var param in fc.parameters()) {
                for (int n = 0; n < param.numel(); n++) {
                    param.data[n] += param.grad[n] * 0.001f;
                }
            }

            if (step % 1000 == 0)
                Console.WriteLine($"step # {step}, loss = {loss}");
        }

        Console.WriteLine($"Loss: {loss:f6} after {step - 1} steps");
        Console.WriteLine($"==> Learned function:\t" + poly_desc(fc._Weight, fc._Bias));
        Console.WriteLine($"==> Actual function:\t" + poly_desc(W_target, b_target));
        Console.ReadKey();
    }
}