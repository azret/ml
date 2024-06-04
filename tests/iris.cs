using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Intrinsics.X86;
using System.Text;

using nn;

using Linear = nn.Linear<nn.CPU.MatMulC>;

internal unsafe class iris {
    static string pretty_logits(float[] logits) {
        fixed (float* logits0 = logits) {
            return pretty_logits(logits0, (uint)logits.Length);
        }
    }

    static string pretty_logits(float* logits0, uint cc, uint max_ = 7) {
        uint n = Math.Min(cc, max_);
        string row = "[";
        for (int j = 0; j < n; j++) {
            row += $"{logits0[j]:f4}";
            if (j == n - 1) {
                if (n < cc)
                    row += ", ...";
            } else {
                row += ", ";
            }
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

    static void run(TextWriter Console, string data_file, string optim, string loss_fn, float lr) {
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

        Console.WriteLine($"fc1.weight: {pretty_logits(fc1._Weight.data, fc1._Weight.numel())}");
        if (fc1._Bias != null) {
            Console.WriteLine($"fc1.bias: {pretty_logits(fc1._Bias.data, fc1._Bias.numel())}");
        }
        Console.WriteLine($"fc2.weight: {pretty_logits(fc2._Weight.data, fc2._Weight.numel())}");
        if (fc2._Bias != null) {
            Console.WriteLine($"fc2.bias: {pretty_logits(fc2._Bias.data, fc2._Bias.numel())}");
        }

        const int batch_size = 10;

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

            Console.WriteLine($"{epoch}: fc2.weight.grad: {pretty_logits(fc2._Weight.grad, fc2._Weight.numel())}");
            if (fc2._Bias != null)
                Console.WriteLine($"{epoch}: fc2.bias.grad: {pretty_logits(fc2._Bias.grad, fc2._Bias.numel())}");
            Console.WriteLine($"{epoch}: fc1.weight.grad: {pretty_logits(fc1._Weight.grad, fc1._Weight.numel())}");
            if (fc1._Bias != null)
                Console.WriteLine($"{epoch}: fc1.bias.grad: {pretty_logits(fc1._Bias.grad, fc1._Bias.numel())}");

            Console.WriteLine($"{epoch}: fc1.weight: {pretty_logits(fc1._Weight.data, fc1._Weight.numel())}");
            if (fc1._Bias != null)
                Console.WriteLine($"{epoch}: fc1.bias: {pretty_logits(fc1._Bias.data, fc1._Bias.numel())}");
            Console.WriteLine($"{epoch}: fc2.weight: {pretty_logits(fc2._Weight.data, fc2._Weight.numel())}");
            if (fc2._Bias != null)
                Console.WriteLine($"{epoch}: fc2.bias: {pretty_logits(fc2._Bias.data, fc2._Bias.numel())}");
        }

        data_iter = get_batch(data, 1).GetEnumerator();

        Console.WriteLine("eval:");

        x.resize(fc1.I);

        for (uint s = 0; s < data.Length; s++) {
            data_iter.MoveNext();
            var sample = data_iter.Current;
            int n = 0;
            for (int i = 0; i < sample.len; i++) {
                for (int j = 0; j < sample.x[i].Length; j++) {
                    x.data[n++] = sample.x[i][j];
                }
            }
            var logits = model.forward(x);
            Console.WriteLine($"{s}: logits: {pretty_logits(logits.data, logits.numel())}");
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

    static int Main() {
        Console.WriteLine("> CPUD: ID: " + GetProcessorManufacturerId());
        Console.WriteLine("> CPUD: X64: " + System.Runtime.Intrinsics.X86.X86Base.X64.IsSupported);
        Console.WriteLine("> CPUD: AVX: " + System.Runtime.Intrinsics.X86.Avx.IsSupported);
        Console.WriteLine("> CPUD: AVX2: " + System.Runtime.Intrinsics.X86.Avx2.IsSupported);
        Console.WriteLine("> CPUD: Avx512F: " + System.Runtime.Intrinsics.X86.Avx512F.IsSupported );
        Console.WriteLine();

        int exitCode = 0;

        Console.WriteLine("> " + Assembly.GetExecutingAssembly().Location);

        string rootPath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        if (!Path.EndsInDirectorySeparator(rootPath)) {
            rootPath += Path.DirectorySeparatorChar;
        }

        Console.WriteLine($"> PATH: {rootPath}");
        Console.WriteLine();

        // =================== Testing RNG =======================

        Console.WriteLine("Testing RNG...");
        var OUT = File.CreateText(rootPath + "iris.csharp.RNG.txt");
        try {
            rng_tests(OUT);
        } catch {
        }
        OUT.Flush();
        OUT.Close();

        OUT = File.CreateText(rootPath + "iris.pytorch.RNG.txt");
        OUT.Write(runpy(rootPath + "bernoulli.py"));
        OUT.Flush();
        OUT.Close();

        if (File.ReadAllText(rootPath + "iris.csharp.RNG.txt") !=
            File.ReadAllText(rootPath + "iris.pytorch.RNG.txt")) {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FAILED!");
            Console.WriteLine("EXPECTED: " + File.ReadAllText(rootPath + "iris.pytorch.RNG.txt"));
            Console.WriteLine();
            Console.WriteLine("ACTUAL: " + File.ReadAllText(rootPath + "iris.csharp.RNG.txt"));
            exitCode = 1;
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK.");
        }
        Console.ResetColor();

        // =================== Testing SGD =======================

        Console.WriteLine("Testing SGD...");
        OUT = File.CreateText(rootPath + "iris.csharp.SGD.txt");
        run(OUT, rootPath + "iris.csv", "SGD", "MSELoss", 1e-3f);
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
            exitCode = 1;
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK.");
        }
        Console.ResetColor();

        // =================== Testing AdamW =======================

        Console.WriteLine("Testing AdamW...");
        OUT = File.CreateText(rootPath + "iris.csharp.AdamW.txt");
        run(OUT, rootPath + "iris.csv", "AdamW", "BCELoss", 1e-6f);
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
            exitCode = 1;
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK.");
        }
        Console.ResetColor();

        if (Debugger.IsAttached) {
            Console.Write("\nPress any key to continue...");
            Console.ReadKey();
        }

        return exitCode;
    }

    static string GetProcessorManufacturerId() {
        if (!X86Base.IsSupported) return null;
        (int _, int ebx, int ecx, int edx) = X86Base.CpuId(0, 0);
        int* manufacturerId = stackalloc int[3] { ebx, edx, ecx };
        return Encoding.ASCII.GetString((byte*)manufacturerId, 12);
    }

    static bool IsGenuineIntel() {
        if (!X86Base.IsSupported) return false;
        (int _, int ebx, int ecx, int edx) = X86Base.CpuId(0, 0);
        return ebx == 0x756e6547  /* Genu */
            && ecx == 0x6c65746e  /* ineI */
            && edx == 0x49656e69  /* ntel */ ;
    }

    private static void rng_tests(TextWriter Console) {
        Console.WriteLine("<bernoulli_>");
        var g = new nn.rand.mt19937(137);
        var a = Tensor.zeros(137);
        Console.WriteLine(g.randint32());
        nn.rand.uniform_(a.data, a.numel(), g, 0, 1);
        Console.WriteLine(pretty_logits(a.data, a.numel()));
        Console.WriteLine(g.randint32());
        bernoulli_(a, 0.2, g);
        Console.WriteLine(pretty_logits(a.data, a.numel(), 137));
        Console.WriteLine(g.randint32());
        Console.WriteLine("</bernoulli_>");
    }

#if MKL
    public enum VslBrng {
        MCG31 = (1 << 20),
        R250 = (1 << 20) * 2,
        MRG32K3A = (1 << 20) * 3,
        MCG59 = (1 << 20) * 4,
        WH = (1 << 20) * 5,
        SOBOL = (1 << 20) * 6,
        NIEDERR = (1 << 20) * 7,
        MT19937 = (1 << 20) * 8,
        MT2203 = (1 << 20) * 9,
        IABSTRACT = (1 << 20) * 10,
        DABSTRACT = (1 << 20) * 11,
        SABSTRACT = (1 << 20) * 12,
        SFMT19937 = (1 << 20) * 13,
        NONDETERM = (1 << 20) * 14,
        ARS5 = (1 << 20) * 15,
        PHILOX4X32X10 = (1 << 20) * 16,
    }

    static void vslCheck(int errorCode) { if (errorCode != 0) throw new InvalidOperationException($"vsl error: {errorCode}"); }

    [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    static extern int vslDeleteStream_64(ref IntPtr stream);

    [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    static extern int vslSkipAheadStream_64(IntPtr stream, ulong nskip);

    [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    static extern int vslNewStream_64(out IntPtr stream, long brng, ulong seed);

    [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    static extern int viRngBernoulli_64(long method, IntPtr stream, long n, int* r, double p);

    [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    static extern int viRngUniform_64(long method, IntPtr stream, long n, int* r, int from, int to);

    [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    static extern int vsRngUniform_64(long method, IntPtr stream, long n, float* r, float from, float to);
    [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    static extern int vdRngUniform_64(long method, IntPtr stream, long n, double[] r, double from, double to);
#endif

#if MKL
    public static void bernoulli_(Tensor a, double p, IRNG g) {
        if (sizeof(float) != sizeof(int)) throw new InvalidProgramException();
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (a.dtype() != DataType.float32) throw new ArgumentException($"Tensor must be of type: {nameof(DataType.float32)}");
        if (IsGenuineIntel()) {
            var seed = g.randint32();
            vslCheck(vslNewStream_64(out var stream, (long)VslBrng.MCG31, seed: seed));
            vslCheck(vslSkipAheadStream_64(stream, 0));
            var r = new int[a.numel()];
            const int VSL_RNG_METHOD_BERNOULLI_ICDF = 0;
            // NB: we reuse the same buffer because sizeof(float32) == sizeof(int32)
            vslCheck(viRngBernoulli_64(VSL_RNG_METHOD_BERNOULLI_ICDF, stream,
                a.numel(),
                (int*)a.data,
                p));
            vslCheck(vslDeleteStream_64(ref stream));
            for (int i = 0; i < a.numel(); i++) {
                // NB: convert back to float32
                a.data[i] = *(int*)(&a.data[i]);
            }
        }
    }
#else
    public static void bernoulli_(Tensor a, double p, IRNG g) {
        if (sizeof(float) != sizeof(int)) throw new InvalidProgramException();
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (a.dtype() != DataType.float32) throw new ArgumentException($"Tensor must be of type: {nameof(DataType.float32)}");
        var seed = g.randint32();
        var MCG31 = new nn.rand.mcg31m1(seed: seed);
        for (int i = 0; i < a.numel(); i++) {
            if (MCG31.randfloat64() < p) {
                a.data[i] = 1f;
            } else {
                a.data[i] = 0f;
            }
        }
    }
#endif

}