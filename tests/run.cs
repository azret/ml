using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.Intrinsics.X86;
using System.Text;

internal unsafe class Run {
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

    public static string runpy(string pyfile) {
        ProcessStartInfo processStartInfo = new ProcessStartInfo();
        processStartInfo.CreateNoWindow = true;
        processStartInfo.RedirectStandardOutput = true;
        processStartInfo.RedirectStandardInput = true;
        processStartInfo.RedirectStandardError = true;
        processStartInfo.UseShellExecute = false;
        processStartInfo.Arguments = pyfile;
        processStartInfo.FileName = "python";
        Process process = new Process();
        process.StartInfo = processStartInfo;
        process.Start();
        process.WaitForExit();
        var output = process.StandardOutput.ReadToEnd();
        var error = process.StandardError.ReadToEnd();
        if (!string.IsNullOrWhiteSpace(error)) {
            Console.WriteLine(error);
        }
        return output;
    }

    static int Main() {
        Console.WriteLine("> cpu: vendor: " + GetProcessorManufacturerId());
        Console.WriteLine("> cpu: x64: " + X86Base.X64.IsSupported);
        Console.WriteLine("> cpu: sse: " + Sse.IsSupported);
        Console.WriteLine("> cpu: sse2: " + Sse2.IsSupported);
        Console.WriteLine("> cpu: sse3: " + Sse3.IsSupported);
        Console.WriteLine("> cpu: avx: " + Avx.IsSupported);
        Console.WriteLine("> cpu: avx2: " + Avx2.IsSupported);
        Console.WriteLine("> cpu: avx512f: " + Avx512F.IsSupported);

        Console.WriteLine();
        Console.WriteLine("> exe: " + Assembly.GetExecutingAssembly().Location);
        string rootPath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        if (!Path.EndsInDirectorySeparator(rootPath)) {
            rootPath += Path.DirectorySeparatorChar;
        }
        Console.WriteLine($"> path: {rootPath}");
        Console.WriteLine();

        int exitCode = 0;

        // =================== Special Tests =======================

        test_bernoulli(rootPath, ref exitCode);

        // =================== Basic Tests =======================

        test_net(rootPath, bias: false, activation: "Identity", optim: "SGD", loss: "MSELoss", lr: 1e-4f, momentum: 0, weight_decay: 0, exitCode: ref exitCode);
        test_net(rootPath, bias: true, activation: "Identity", optim: "AdamW", loss: "MSELoss", lr: 1e-4f, momentum: 0, weight_decay: 0, exitCode: ref exitCode, decimals: 3);
        test_net(rootPath, bias: false, activation: "Identity", optim: "AdamW", loss: "MSELoss", lr: 1e-4f, momentum: 0, weight_decay: 0, exitCode: ref exitCode, decimals: 3);
        test_net(rootPath, bias: true, activation: "Identity", optim: "SGD", loss: "MSELoss", lr: 1e-4f, momentum: 0, weight_decay: 0, exitCode: ref exitCode);

        test_net(rootPath, bias: false, activation: "ReLU", optim: "SGD", loss: "MSELoss", lr: 1e-3f, momentum: 0, weight_decay: 0, exitCode: ref exitCode);
        test_net(rootPath, bias: false, activation: "Sigmoid", optim: "SGD", loss: "MSELoss", lr: 1e-3f, momentum: 0, weight_decay: 0, exitCode: ref exitCode);
        test_net(rootPath, bias: false, activation: "LeakyReLU", optim: "SGD", loss: "MSELoss", lr: 1e-3f, momentum: 0, weight_decay: 0, exitCode: ref exitCode);
        test_net(rootPath, bias: false, activation: "Tanh", optim: "SGD", loss: "MSELoss", lr: 1e-3f, momentum: 0, weight_decay: 0, exitCode: ref exitCode);
        test_net(rootPath, bias: false, activation: "Dropout", optim: "SGD", loss: "MSELoss", lr: 1e-3f, momentum: 0, weight_decay: 0, exitCode: ref exitCode);

        test_net(rootPath, bias: true, activation: "Identity", optim: "SGD", loss: "MSELoss", lr: 1e-1f, momentum: 0, weight_decay: 0, exitCode: ref exitCode);
        test_net(rootPath, bias: false, activation: "Identity", optim: "SGD", loss: "MSELoss", lr: 1e-1f, momentum: 1e-1f, weight_decay: 1e-5f, exitCode: ref exitCode);
        test_net(rootPath, bias: true, activation: "Identity", optim: "SGD", loss: "MSELoss", lr: 1e-1f, momentum: 0, weight_decay: 1e-5f, exitCode: ref exitCode);
        test_net(rootPath, bias: false, activation: "Identity", optim: "SGD", loss: "MSELoss", lr: 1e-1f, momentum: 1e-1f, weight_decay: 0, exitCode: ref exitCode);

        test_net(rootPath, bias: true, activation: "Identity", optim: "AdamW", loss: "MSELoss", lr: 1e-4f, momentum: 0, weight_decay: 0, exitCode: ref exitCode, decimals: 3);
        test_net(rootPath, bias: true, activation: "Identity", optim: "AdamW", loss: "MSELoss", lr: 1e-4f, momentum: 0, weight_decay: 1e-5f, exitCode: ref exitCode, decimals: 3);
        test_net(rootPath, bias: false, activation: "Identity", optim: "AdamW", loss: "MSELoss", lr: 1e-4f, momentum: 0, weight_decay: 0, exitCode: ref exitCode, decimals: 3);
        test_net(rootPath, bias: false, activation: "Identity", optim: "AdamW", loss: "MSELoss", lr: 1e-4f, momentum: 0, weight_decay: 1e-5f, exitCode: ref exitCode, decimals: 3);

        test_net(rootPath, bias: false, activation: "Identity", optim: "AdamW", loss: "BCELoss", lr: 1e-5f, momentum: 0, weight_decay: 0, exitCode: ref exitCode, decimals: 3);
        test_net(rootPath, bias: false, activation: "Identity", optim: "SGD", loss: "BCELoss", lr: 1e-5f, momentum: 0, weight_decay: 0, exitCode: ref exitCode);

        if (Debugger.IsAttached) {
            Console.Write("\nPress any key to continue...");
            Console.ReadKey();
        }

        return exitCode;
    }

    private static void test_net(
        string rootPath,
        bool bias,
        string activation,
        string optim,
        string loss,
        float lr,
        float momentum,
        float weight_decay,
        ref int exitCode,
        int decimals = 4) {

        if (string.IsNullOrWhiteSpace(rootPath)) {
            throw new ArgumentException($"'{nameof(rootPath)}' cannot be null or whitespace.", nameof(rootPath));
        }
        if (string.IsNullOrWhiteSpace(activation)) {
            throw new ArgumentException($"'{nameof(activation)}' cannot be null or whitespace.", nameof(activation));
        }

        string cs_log = "test_net.cs.txt";
        string py_log = "test_net.py.txt";

        string py_args = $"generic.py --d {decimals} --loss {loss} --bias {(bias ? "yes" : "no")} --a {activation} --o {optim} --lr {lr:0e+00} --m {momentum:0e+00} --weight_decay {weight_decay:0e+00}";

        if (File.Exists(rootPath + cs_log)) File.Delete(rootPath + cs_log);
        if (File.Exists(rootPath + py_log)) File.Delete(rootPath + py_log);

        Console.WriteLine(py_args);

        StreamWriter OUT = File.CreateText(rootPath + cs_log);
        generic.test_net(OUT, bias, activation, optim, loss, lr, momentum, weight_decay, decimals);
        OUT.Flush();
        OUT.Close();

        OUT = File.CreateText(rootPath + py_log);
        OUT.Write(runpy(rootPath + py_args));
        OUT.Flush();
        OUT.Close();

        if (File.ReadAllText(rootPath + cs_log) !=
            File.ReadAllText(rootPath + py_log)) {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("ERROR!");
                var py_lines = File.ReadAllLines(rootPath + py_log);
                var cs_lines = File.ReadAllLines(rootPath + cs_log);
                for (int i = 0; i < py_lines.Length; i++) {
                    if (py_lines[i] != cs_lines[i]) {
                        Console.ForegroundColor = ConsoleColor.DarkYellow;
                        Console.WriteLine("\nEXPECTED: " + py_lines[i]);
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine("ACTUAL: " + cs_lines[i]);
                    }
                }
                exitCode = 1;
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
        }

        Console.ResetColor();
    }

    static void test_bernoulli(string rootPath, ref int exitCode) {
        string cs_log = "test_bernoulli.cs.txt";
        string py_log = "test_bernoulli.py.txt";

        Console.WriteLine($"{nameof(test_bernoulli)}");
        var OUT = File.CreateText(rootPath + cs_log);
        try {
            bernoulli.test_bernoulli(OUT);
        } catch {
        }
        OUT.Flush();
        OUT.Close();

        OUT = File.CreateText(rootPath + py_log);
        OUT.Write(runpy(rootPath + "bernoulli.py"));
        OUT.Flush();
        OUT.Close();

        if (File.ReadAllText(rootPath + cs_log) !=
            File.ReadAllText(rootPath + py_log)) {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FAILED!");
            Console.WriteLine("\nEXPECTED: " + File.ReadAllText(rootPath + py_log));
            Console.WriteLine();
            Console.WriteLine("ACTUAL: " + File.ReadAllText(rootPath + cs_log));
            exitCode = 1;
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK.");
        }
        Console.ResetColor();
    }
}