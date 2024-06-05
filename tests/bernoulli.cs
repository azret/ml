using System;
using System.IO;

unsafe internal static class bernoulli {
    public static void test_bernoulli(TextWriter Console) {
        Console.WriteLine("<bernoulli_>");
        var g = new nn.rand.mt19937(137);
        var a = nn.Tensor.zeros(137);
        Console.WriteLine(g.randint32());
        nn.rand.uniform_(a.data, a.numel(), g, 0, 1);
        Console.WriteLine(Common.pretty_logits(a.data, a.numel()));
        Console.WriteLine(g.randint32());
        bernoulli_(a, 0.2, g);
        Console.WriteLine(Common.pretty_logits(a.data, a.numel(), 137));
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
    public static void bernoulli_(nn.Tensor a, double p, nn.IRNG g) {
        if (a == null) throw new ArgumentNullException(nameof(a));
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