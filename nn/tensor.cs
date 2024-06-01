namespace nn {
    using System;
    using System.Diagnostics;
    using System.Runtime.ConstrainedExecution;
    using System.Threading;

    using static std;

    public enum Device {
        CPU,
        CUDA
    }

    [DebuggerDisplay("{global::std.memsize(numbytes)} ({device})")]
    public unsafe sealed partial class Tensor : CriticalFinalizerObject, IDisposable {
        public static Tensor zeros(uint numel, bool requires_grad = false) {
            return new Tensor(numel, requires_grad);
        }

        public static Tensor ones(uint numel, bool requires_grad = false) {
            var T = new Tensor(numel, requires_grad);
            T.fill_(1f);
            return T;
        }

        public static Tensor from(float[] other, bool requires_grad = false) {
            var T = new Tensor((uint)other.Length, requires_grad);
            for (uint t = 0; t < T.numel(); t++) {
                T.data[t] = other[t];
            }
            return T;
        }

        const ulong ALIGNMENT = 4096ul;

        uint _numel;

        public Tensor(uint numel, bool requires_grad = true) : base() {
            alloc_cpu(numel, requires_grad, out data, out grad);
        }

        ~Tensor() {
            Dispose();
        }

        public void Dispose() {
            free_cpu();
            GC.SuppressFinalize(this);
        }

        public readonly float* data;

        public readonly float* grad;

        public readonly Device device;

        IntPtr h_ua_data, h_ua_grad;

        void alloc_cpu(uint numel, bool requires_grad, out float* data, out float* grad) {
            try {
                _numel = numel; data = null; grad = null;
                if (_numel > 0) {
                    h_ua_data = (IntPtr)malloc(ALIGNMENT + (ulong)_numel * sizeof(float));
                    memset((void*)h_ua_data, 0, ALIGNMENT + sizeof(float) * (ulong)_numel);
                }
                if (_numel > 0 && requires_grad) {
                    h_ua_grad = (IntPtr)malloc(ALIGNMENT + (ulong)_numel * sizeof(float));
                    memset((void*)h_ua_grad, 0, ALIGNMENT + sizeof(float) * (ulong)_numel);
                }
                // We need to ensure that the host memory is aligned to 4K
                if (h_ua_data != IntPtr.Zero) data = (float*)(((ulong)h_ua_data + (ALIGNMENT - 1)) & (~(ALIGNMENT - 1)));
                if (h_ua_grad != IntPtr.Zero) grad = (float*)(((ulong)h_ua_grad + (ALIGNMENT - 1)) & (~(ALIGNMENT - 1)));
            } catch {
                data = null;
                grad = null;
                _numel = 0;
                if (h_ua_grad != IntPtr.Zero) global::std.free((void*)h_ua_grad);
                if (h_ua_data != IntPtr.Zero) global::std.free((void*)h_ua_data);
                throw;
            }
        }

        void free_cpu() {
            free((void*)Interlocked.Exchange(ref h_ua_data, IntPtr.Zero));
            free((void*)Interlocked.Exchange(ref h_ua_grad, IntPtr.Zero));
        }

        public ulong numbytes {
            get {
                ulong numbytes = 0;
                if (h_ua_data != IntPtr.Zero) {
                    numbytes += ALIGNMENT + (ulong)_numel * sizeof(float);
                }
                if (h_ua_grad != IntPtr.Zero) {
                    numbytes += ALIGNMENT + (ulong)_numel * sizeof(float);
                }
                return numbytes;
            }
        }

        public uint numel() {
            return _numel;
        }

        public void resize(uint value) {
            if (_numel > value) {
                _numel = value;
            } else {
                throw new NotImplementedException();
            }
        }

        public static double pdf(double x, double mean, double std) {
            var ω = (x - mean) / std;
            return Math.Exp(-0.5 * ω * ω)
                / (2.5066282746310005024157652848110452530069867406099d * std);
        }

        public double min() {
            var min = double.PositiveInfinity;
            for (uint t = 0; t < numel(); t++) {
                if (data[t] < min)
                    min = data[t];
            }
            return min;
        }
        public double max() {
            var max = double.NegativeInfinity;
            for (uint t = 0; t < numel(); t++) {
                if (data[t] > max)
                    max = data[t];
            }
            return max;
        }
        public double mean() {
            var mean = 0.0d;
            for (uint t = 0; t < numel(); t++) {
                mean += data[t];
            }
            return (mean / numel());
        }
        public double variance() {
            return variance(mean());
        }
        public double variance(double mean) {
            var variance = 0.0d;
            for (uint t = 0; t < numel(); t++) {
                variance += Math.Pow(data[t] - mean, 2);
            }
            return (variance / numel());
        }
        public double std() {
            return (Math.Sqrt(variance()));
        }
        public double std(double variance) {
            return (Math.Sqrt(variance));
        }

        public void sin_(float w, Func<uint, uint, float> env = null) {
            for (uint t = 0; t < numel(); t++) {
                data[t] = (float)Math.Sin(t * w);
                if (env != null) {
                    data[t] *= env(t, numel());
                }
            }
        }

        public void cos_(float w, Func<uint, uint, float> env = null) {
            for (uint t = 0; t < numel(); t++) {
                data[t] = (float)Math.Cos(t * w);
                if (env != null) {
                    data[t] *= env(t, numel());
                }
            }
        }

        public void fill_(float fill) {
            for (uint t = 0; t < numel(); t++) {
                data[t] = fill;
            }
        }

        public void linspace_(float from, float to, bool endpoint = true) {
            if (endpoint) {
                if (numel() == 1) {
                    data[0] = from;
                } else {
                    float dx = (to - from) / (numel() - 1);
                    for (int t = 0; t < numel(); t++) {
                        data[t] = from + dx * t;
                    }
                }
            } else {
                float dx = (to - from) / (numel());
                for (int t = 0; t < numel(); t++) {
                    data[t] = from + dx * t;
                }
            }
        }

        public void logspace_(float from, float to, bool endpoint = true, float logBase = 10) {
            linspace_((float)Math.Log(from, logBase), (float)Math.Log(to, logBase), endpoint);
            for (uint t = 0; t < numel(); t++) {
                data[t] = (float)Math.Pow(logBase, data[t]);
            }
        }

        public void memcpy(float* src, uint numel) {
            kernel32.CopyMemory(
                data,
                src,
                (UIntPtr)((ulong)numel * sizeof(float)));
        }

        public void zero_data() {
            kernel32.ZeroMemory(
                data,
                (UIntPtr)((ulong)numel() * sizeof(float)));
        }

        public void zero_grad() {
            kernel32.ZeroMemory(
                grad,
                (UIntPtr)((ulong)numel() * sizeof(float)));
        }
    }
}