using System.Collections.Generic;
using System.Diagnostics;
using System;
using System.IO;
using System.Linq;

using nn;

unsafe internal static class iris {
    static IEnumerable<(float[][] x, float[][] y, uint B)> get_batch(string[] data, uint B) {
        int i = 0;
        while (true) {
            int take = checked((int)B);
            if (i + take > data.Length)
                take = data.Length - i;
            var x = data.Skip(i).Take(take).ToArray().Select(
                (s) => {
                    var split = s.Split(',').Take(4)
                        .Select(k => float.Parse(k)).ToArray();
                    return split;
                }).ToArray();
            var y = data.Skip(i).Take(take).ToArray().Select(
                (s) => {
                    var split = s.Split(',').Skip(4).Take(3)
                        .Select(k => float.Parse(k)).ToArray();
                    return split;
                }).ToArray();
            Debug.Assert(x.Length == take);
            Debug.Assert(y.Length == take);
            yield return (x, y, (uint)take);
            i += take;
            if (i >= data.Length)
                i = 0;
        }
    }

    static void reset_weights(Linear lin, IRNG g) {
        nn.init.kaiming_uniform_(
            lin._Weight.data,
            lin._Weight.numel(),
            g,
            lin.I,
            (float)Math.Sqrt(5));

        if (lin._Bias != null) {
            nn.rand.uniform_(
                lin._Bias.data,
                lin._Bias.numel(),
                g,
                -(float)(1.0 / Math.Sqrt(lin.I)),
                (float)(1.0 / Math.Sqrt(lin.I)));
        }
    }

    public static void run(TextWriter Console, string data_file, string optim, string loss_fn, float lr, uint batch_size) {

        var data = File.ReadAllLines(data_file);

        // test data loader batching

        // var data_iter_test = get_batch(data, 37).GetEnumerator();
        // data_iter_test.MoveNext();
        // for (int i = 0; i < 5; i++) {
        //     var c = data_iter_test.Current;
        //     Console.WriteLine($"batch_size: {c.len}:");
        //     for (int j = 0; j < c.len; j++) {
        //         Console.WriteLine($"x: [{string.Join(", ", c.x[j].Select(f => $"{f:f4}"))}] = y: [{string.Join(", ", c.y[j].Select(f => $"{f:f4}"))}]");
        //     }
        //     data_iter_test.MoveNext();
        // }

        var g = new nn.rand.mt19937(137);

        Linear fc1, fc2;

        nn.Sequential model = new nn.Sequential(
            fc1 = new Linear(4, 8, bias: true),
            new nn.Identity(),
            new nn.ReLU(),
            fc2 = new Linear(8, 3, bias: false),
            new nn.Identity(),
            new nn.Sigmoid()
        );

        reset_weights(fc1, g);
        reset_weights(fc2, g);

        Console.WriteLine($"fc1.weight: {Common.pretty_logits(fc1._Weight.data, fc1._Weight.numel())}");
        if (fc1._Bias != null) {
            Console.WriteLine($"fc1.bias: {Common.pretty_logits(fc1._Bias.data, fc1._Bias.numel())}");
        }
        Console.WriteLine($"fc2.weight: {Common.pretty_logits(fc2._Weight.data, fc2._Weight.numel())}");
        if (fc2._Bias != null) {
            Console.WriteLine($"fc2.bias: {Common.pretty_logits(fc2._Bias.data, fc2._Bias.numel())}");
        }

        var x = Tensor.zeros(batch_size * fc1.I);
        var y = new float[batch_size * fc2.O];

        for (int i = 0; i < y.Length; i++) {
            y[i] = float.NaN;
        }

        IOptimizer optimizer = null;

        if (optim == "SGD")
            optimizer = new nn.SGD(model.parameters(), lr: lr);
        else if (optim == "AdamW")
            optimizer = new nn.AdamW(model.parameters(), lr: lr);

        Console.WriteLine($"parameters: {optimizer.get_num_params()}");

        int epochs = 10;

        var data_iter = get_batch(data, batch_size).GetEnumerator();

        Console.WriteLine("train:");

        for (uint epoch = 0; epoch < epochs; epoch++) {
            data_iter.MoveNext();
            var sample = data_iter.Current;

            Console.WriteLine($"batch_size: {sample.B}");

            int n = 0;

            x.resize(sample.B * fc1.I);

            for (int i = 0; i < sample.B; i++) {
                for (int j = 0; j < sample.x[i].Length; j++) {
                    x.data[n++] = sample.x[i][j];
                }
            }

            Console.WriteLine($"x: {Common.pretty_logits(x.data, x.numel())}");

            n = 0;

            for (int i = 0; i < sample.B; i++) {
                for (int j = 0; j < sample.y[i].Length; j++) {
                    y[n++] = sample.y[i][j];
                }
            }

            var start_time = kernel32.millis();

            var logits = model.forward(x);

            Console.WriteLine($"{epoch}: logits: {Common.pretty_logits(logits.data, logits.numel())}");

            double loss = double.NaN;

            if (loss_fn == "BCELoss")
                loss = F.binary_cross_entropy(
                    logits,
                    y);
            else if (loss_fn == "MSELoss")
                loss = F.mse_loss(
                    logits.data,
                    logits.grad,
                    logits.numel(),
                    y);

            Console.WriteLine($"{epoch}: loss: {loss:f4}");

            optimizer.zero_grad();

            model.backward(logits);

            var elapsedMillis = (kernel32.millis() - start_time);

            optimizer.step();

            Console.WriteLine($"{epoch}: fc2.weight.grad: {Common.pretty_logits(fc2._Weight.grad, fc2._Weight.numel())}");
            if (fc2._Bias != null)
                Console.WriteLine($"{epoch}: fc2.bias.grad: {Common.pretty_logits(fc2._Bias.grad, fc2._Bias.numel())}");
            Console.WriteLine($"{epoch}: fc1.weight.grad: {Common.pretty_logits(fc1._Weight.grad, fc1._Weight.numel())}");
            if (fc1._Bias != null)
                Console.WriteLine($"{epoch}: fc1.bias.grad: {Common.pretty_logits(fc1._Bias.grad, fc1._Bias.numel())}");

            Console.WriteLine($"{epoch}: fc1.weight: {Common.pretty_logits(fc1._Weight.data, fc1._Weight.numel())}");
            if (fc1._Bias != null)
                Console.WriteLine($"{epoch}: fc1.bias: {Common.pretty_logits(fc1._Bias.data, fc1._Bias.numel())}");
            Console.WriteLine($"{epoch}: fc2.weight: {Common.pretty_logits(fc2._Weight.data, fc2._Weight.numel())}");
            if (fc2._Bias != null)
                Console.WriteLine($"{epoch}: fc2.bias: {Common.pretty_logits(fc2._Bias.data, fc2._Bias.numel())}");
        }

        data_iter = get_batch(data, 1).GetEnumerator();

        Console.WriteLine("eval:");

        x.resize(fc1.I);

        for (uint s = 0; s < data.Length; s++) {
            data_iter.MoveNext();
            var sample = data_iter.Current;
            int n = 0;
            Debug.Assert(sample.B == 1);
            for (int j = 0; j < sample.x[0].Length; j++) {
                x.data[n++] = sample.x[0][j];
            }
            var logits = model.forward(x);
            Console.WriteLine($"{s}: logits: {Common.pretty_logits(logits.data, logits.numel())}");
        }
    }
}