using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

using nn;

internal unsafe class iris {
    static string pretty_logits(float[] logits) {
        fixed (float* logits0 = logits) {
            return pretty_logits(logits0, (uint)logits.Length);
        }
    }

    static string pretty_logits(float* logits0, uint cc) {
        string row = "[";
        for (int j = 0; j < Math.Min(cc, 7); j++) {
            row += $"{(Math.Round(logits0[j], 4)):f4}";
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

    public static IEnumerable<(float[][] x, float[][] y, int len)> get_batch(string[] data, int B) {
        int i = 0;
        while (true) {
            int take = B;
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
            yield return (x, y, take);
            i += take;
            if (i >= data.Length)
                i = 0;
        }
    }

    private static void reset_weights(Linear lin, IRNG g) {
        nn.rand.kaiming_uniform_(
            lin.weight.data,
            lin.weight.numel(),
            g,
            lin._in_features,
            (float)Math.Sqrt(5));

        if (lin.bias != null) {
            nn.rand.uniform_(
                lin.bias.data,
                lin.bias.numel(),
                g,
                -(float)(1.0 / Math.Sqrt(lin._in_features)),
                (float)(1.0 / Math.Sqrt(lin._in_features)));
        }
    }

    static void run(TextWriter Console, string optim, string loss_fn, float lr) {
        var data = File.ReadAllLines("iris.csv");

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

        nn.Linear fc1, fc2;

        nn.Sequential model = new nn.Sequential(
            fc1 = new nn.Linear(4, 8, use_bias: true),
            new nn.Identity(),
            new nn.ReLU(),
            fc2 = new nn.Linear(8, 3, use_bias: false),
            new nn.Identity(),
            new nn.Sigmoid()
        );

        reset_weights(fc1, g);
        reset_weights(fc2, g);

        Console.WriteLine($"fc1.weight: {pretty_logits(fc1.weight.data, fc1.weight.numel())}");
        if (fc1.bias != null) {
            Console.WriteLine($"fc1.bias: {pretty_logits(fc1.bias.data, fc1.bias.numel())}");
        }
        Console.WriteLine($"fc2.weight: {pretty_logits(fc2.weight.data, fc2.weight.numel())}");
        if (fc2.bias != null) {
            Console.WriteLine($"fc2.bias: {pretty_logits(fc2.bias.data, fc2.bias.numel())}");
        }

        const int batch_size = 10;

        var x = Tensor.zeros(batch_size * 4);
        var y = new float[batch_size * 3];
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

        for (uint epoch = 0; epoch < epochs; epoch++) {
            data_iter.MoveNext();
            var sample = data_iter.Current;

            int n = 0;
            for (int i = 0; i < sample.len; i++) {
                for (int j = 0; j < sample.x[i].Length; j++) {
                    x.data[n++] = sample.x[i][j];
                }
            }

            n = 0;
            for (int i = 0; i < sample.len; i++) {
                for (int j = 0; j < sample.y[i].Length; j++) {
                    y[n++] = sample.y[i][j];
                }
            }

            var start_time = kernel32.millis();

            var logits = model.forward(x);
            Console.WriteLine($"{epoch}: logits: {pretty_logits(logits.data, logits.numel())}");

            double loss = double.NaN;
            if (loss_fn == "BCELoss")
                loss = F.binary_cross_entropy(logits, y);
            else if (loss_fn == "MSELoss")
                loss = F.mse_loss(logits.data, logits.grad, y);

            Console.WriteLine($"{epoch}: loss: {loss:f4}");

            optimizer.zero_grad();

            model.backward(logits);

            var elapsedMillis = (kernel32.millis() - start_time);

            optimizer.step();

            Console.WriteLine($"{epoch}: fc2.weight.grad: {pretty_logits(fc2.weight.grad, fc2.weight.numel())}");
            if (fc2.bias != null)
                Console.WriteLine($"{epoch}: fc2.bias.grad: {pretty_logits(fc2.bias.grad, fc2.bias.numel())}");
            Console.WriteLine($"{epoch}: fc1.weight.grad: {pretty_logits(fc1.weight.grad, fc1.weight.numel())}");
            if (fc1.bias != null)
                Console.WriteLine($"{epoch}: fc1.bias.grad: {pretty_logits(fc1.bias.grad, fc1.bias.numel())}");

            Console.WriteLine($"{epoch}: fc1.weight: {pretty_logits(fc1.weight.data, fc1.weight.numel())}");
            if (fc1.bias != null)
                Console.WriteLine($"{epoch}: fc1.bias: {pretty_logits(fc1.bias.data, fc1.bias.numel())}");
            Console.WriteLine($"{epoch}: fc2.weight: {pretty_logits(fc2.weight.data, fc2.weight.numel())}");
            if (fc2.bias != null)
                Console.WriteLine($"{epoch}: fc2.bias: {pretty_logits(fc2.bias.data, fc2.bias.numel())}");
        }
    }

    public static string runpy(string pyfile) {
        ProcessStartInfo processStartInfo = new ProcessStartInfo();
        processStartInfo.CreateNoWindow = true;
        processStartInfo.RedirectStandardOutput = true;
        processStartInfo.RedirectStandardInput = true;
        processStartInfo.UseShellExecute = false;
        processStartInfo.Arguments = pyfile;
        processStartInfo.FileName = "python";
        Process process = new Process();
        process.StartInfo = processStartInfo;
        process.Start();
        var output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        return output;
    }

    static void Main() {
        string rootPath = "D:\\SRC\\nn.cs\\tests\\";

        Console.WriteLine("Testing SGD...");
        var OUT = File.CreateText(rootPath + "iris.csharp.SGD.txt");
        run(OUT, "SGD", "MSELoss", 1e-3f);
        OUT.Flush();
        OUT.Close();

        OUT = File.CreateText(rootPath + "iris.pytorch.SGD.txt");
        OUT.Write(runpy(rootPath + "iris.py --batch_size 10 --optim SGD --lr 1e-3 --loss MSELoss"));
        OUT.Flush();
        OUT.Close();

        if (File.ReadAllText(rootPath + "iris.csharp.SGD.txt") !=
                File.ReadAllText(rootPath + "iris.pytorch.SGD.txt")) {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FAILED!");
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK.");
        }
        Console.ResetColor();

        Console.WriteLine("Testing AdamW...");
        OUT = File.CreateText(rootPath + "iris.csharp.AdamW.txt");
        run(OUT, "AdamW", "BCELoss", 1e-6f);
        OUT.Flush();
        OUT.Close();

        OUT = File.CreateText(rootPath + "iris.pytorch.AdamW.txt");
        OUT.Write(runpy(rootPath + "iris.py --batch_size 10 --optim AdamW --lr 1e-6 --loss BCELoss"));
        OUT.Flush();
        OUT.Close();

        if (File.ReadAllText(rootPath + "iris.csharp.AdamW.txt") !=
        File.ReadAllText(rootPath + "iris.pytorch.AdamW.txt")) {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FAILED!");
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK.");
        }
        Console.ResetColor();

        Console.WriteLine("Press any key to continue...");
        Console.ReadKey();
    }
}