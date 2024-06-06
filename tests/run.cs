using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.Intrinsics.X86;
using System.Text;

using nn;

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
        Console.WriteLine("> cpu: vendor: " + GetProcessorManufacturerId());
        Console.WriteLine("> cpu: x64: " + System.Runtime.Intrinsics.X86.X86Base.X64.IsSupported);
        Console.WriteLine("> cpu: sse: " + System.Runtime.Intrinsics.X86.Sse.IsSupported);
        Console.WriteLine("> cpu: sse2: " + System.Runtime.Intrinsics.X86.Sse2.IsSupported);
        Console.WriteLine("> cpu: sse3: " + System.Runtime.Intrinsics.X86.Sse3.IsSupported);
        Console.WriteLine("> cpu: avx: " + System.Runtime.Intrinsics.X86.Avx.IsSupported);
        Console.WriteLine("> cpu: avx2: " + System.Runtime.Intrinsics.X86.Avx2.IsSupported);
        Console.WriteLine("> cpu: avx512f: " + System.Runtime.Intrinsics.X86.Avx512F.IsSupported);
        Console.WriteLine();
        Console.WriteLine("> exe: " + Assembly.GetExecutingAssembly().Location);
        string rootPath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        if (!Path.EndsInDirectorySeparator(rootPath)) {
            rootPath += Path.DirectorySeparatorChar;
        }
        Console.WriteLine($"> path: {rootPath}");
        Console.WriteLine();

        int exitCode = 0;

        // =================== Testing RNG =======================

        test_bernoulli(rootPath, ref exitCode);

        // =================== Testing Dropout =======================

        test_dropout(rootPath, ref exitCode);

        // =================== Testing SGD =======================

        test_sgd(rootPath, ref exitCode);

        // =================== Testing AdamW =======================

        Console.WriteLine("Testing AdamW...");
        StreamWriter OUT = File.CreateText(rootPath + "iris.csharp.AdamW.txt");
        iris.test_iris(OUT, rootPath + "iris.csv", "AdamW", "BCELoss", 1e-6f, batch_size: 30, maxDegreeOfParallelism: 0);
        OUT.Flush();
        OUT.Close();

        OUT = File.CreateText(rootPath + "iris.pytorch.AdamW.txt");
        OUT.Write(runpy(rootPath + "iris.py --batch_size 30 --optim AdamW --lr 1e-6 --loss BCELoss"));
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

    private static void test_sgd(string rootPath, ref int exitCode) {
        Console.WriteLine("Testing SGD...");

        var OUT = File.CreateText(rootPath + "iris.csharp.SGD.txt");
        iris.test_iris(OUT, rootPath + "iris.csv", "SGD", "MSELoss", 1e-3f, batch_size: 40, maxDegreeOfParallelism: 0);
        OUT.Flush();
        OUT.Close();

        OUT = File.CreateText(rootPath + "iris.pytorch.SGD.txt");
        OUT.Write(runpy(rootPath + "iris.py --batch_size 40 --optim SGD --lr 1e-3 --loss MSELoss"));
        OUT.Flush();
        OUT.Close();

        if (File.ReadAllText(rootPath + "iris.csharp.SGD.txt") !=
                File.ReadAllText(rootPath + "iris.pytorch.SGD.txt")) {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("ERROR!");
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine("\nEXPECTED: " + File.ReadAllText(rootPath + "iris.pytorch.SGD.txt"));
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("ACTUAL: " + File.ReadAllText(rootPath + "iris.csharp.SGD.txt"));
            exitCode = 1;
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK.");
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

    static void test_dropout_forward(string rootPath, double p, string mode, ref int exitCode) {
        string cs_log = "cs.test_dropout_forward.txt";
        string py_log = "py.test_dropout_forward.txt";

        Console.WriteLine($"{nameof(test_dropout_forward)} w/ p = {p}, mode = {mode}");
        var OUT = File.CreateText(rootPath + cs_log);
        if (mode == "train")
            dropout.test_dropout_forward(OUT, p, train: true);
        else if (mode == "eval")
            dropout.test_dropout_forward(OUT, p, train: false);
        else
            throw new ArgumentOutOfRangeException(nameof(mode));
        OUT.Flush();
        OUT.Close();
        OUT = File.CreateText(rootPath + py_log);
        OUT.Write(runpy(rootPath + "dropout.py " + "--p=" + p + " --mode=" + mode));
        OUT.Flush();
        OUT.Close();
        if (File.ReadAllText(rootPath + cs_log) !=
            File.ReadAllText(rootPath + py_log)) {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FAILED!");
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine("\nEXPECTED: " + File.ReadAllText(rootPath + py_log));
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("ACTUAL: " + File.ReadAllText(rootPath + cs_log));
            exitCode = 1;
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK.");
        }
        Console.ResetColor();
    }

    static void test_dropout_backward(string rootPath, double p, string use_dropout, ref int exitCode) {
        string cs_log = "test_dropout_backward.cs.txt";
        string py_log = "test_dropout_backward.py.txt";

        Console.WriteLine($"{nameof(test_dropout_backward)} w/ p = {p}, dropout = {use_dropout}");
        var OUT = File.CreateText(rootPath + cs_log);
        if (use_dropout == "yes")
            dropout.test_dropout_backward(OUT, p, use_dropout: true);
        else if (use_dropout == "no")
            dropout.test_dropout_backward(OUT, p, use_dropout: false);
        else
            throw new ArgumentOutOfRangeException(nameof(use_dropout));
        OUT.Flush();
        OUT.Close();
        OUT = File.CreateText(rootPath + py_log);
        OUT.Write(runpy(rootPath + "dropout.py --net=yes --p=" + p + " --dropout=" + use_dropout));
        OUT.Flush();
        OUT.Close();
        if (File.ReadAllText(rootPath + cs_log) !=
            File.ReadAllText(rootPath + py_log)) {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("FAILED!");
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine("\nEXPECTED: " + File.ReadAllText(rootPath + py_log));
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("ACTUAL: " + File.ReadAllText(rootPath + cs_log));
            exitCode = 1;
        } else {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK.");
        }
        Console.ResetColor();
    }

    static void test_dropout(string rootPath, ref int exitCode) {
        test_dropout_forward(rootPath, 0, "train", ref exitCode);
        test_dropout_forward(rootPath, 0.3, "train", ref exitCode);
        test_dropout_forward(rootPath, 1, "train", ref exitCode);

        test_dropout_forward(rootPath, 0, "eval", ref exitCode);
        test_dropout_forward(rootPath, 0.3, "eval", ref exitCode);
        test_dropout_forward(rootPath, 1, "eval", ref exitCode);

        test_dropout_backward(rootPath, 0, "yes", ref exitCode);
        test_dropout_backward(rootPath, 0.3, "yes", ref exitCode);
        test_dropout_backward(rootPath, 1, "yes", ref exitCode);

        test_dropout_backward(rootPath, 0, "no", ref exitCode);
        test_dropout_backward(rootPath, 0.3, "no", ref exitCode);
        test_dropout_backward(rootPath, 1, "no", ref exitCode);
    }
}