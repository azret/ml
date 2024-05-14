namespace nn {
    using System;
    using System.Diagnostics;
    using System.Runtime.ConstrainedExecution;
    using System.Threading;

    using static std;

    [DebuggerDisplay("{Tensor.memsize(numbytes)}")]
    public unsafe sealed partial class Tensor : CriticalFinalizerObject, IDisposable {
        public static Tensor zeros(uint numel, bool requires_grad = false) {
            return new Tensor(numel, requires_grad);
        }

        public static Tensor ones(uint numel, bool requires_grad = false) {
            var tensor = new Tensor(numel, requires_grad);
            tensor.fill_(1f);
            return tensor;
        }

        public static string memsize(ulong size) {
            string[] sizes = { "B", "KiB", "MiB", "GiB", "TiB" };
            double len = size;
            int order = 0;
            while (len >= 1024 && order < sizes.Length - 1) {
                order++;
                len = len / 1024;
            }
            return string.Format("{0:0.##} {1}", len, sizes[order]);
        }

        const ulong ALIGNMENT = 4096ul;

        uint _numel;

        IntPtr h_data, h_grad;

        public readonly float* data, grad;

        public Tensor(uint numel, bool requires_grad = true)
            : base() {
            _numel = numel;
            Alloc(requires_grad, out data, out grad);
        }

        public Tensor(float[] input, bool requires_grad = true)
            : base() {
            _numel = (uint)input.Length;
            Alloc(requires_grad, out data, out grad);
            for (uint t = 0; t < _numel; t++) {
                data[t] = input[t];
            }
        }

        ~Tensor() {
            Dispose();
        }

        public void Dispose() {
            Free();
            GC.SuppressFinalize(this);
        }

        void Alloc(bool requires_grad, out float* data, out float* grad) {
            data = null; grad = null;
            if (_numel > 0) {
                h_data = (IntPtr)malloc(ALIGNMENT + (ulong)_numel * sizeof(float));
                memset((void*)h_data,
                    0,
                    ALIGNMENT + sizeof(float) * (ulong)_numel);
            }
            if (_numel > 0 && requires_grad) {
                h_grad = (IntPtr)malloc(ALIGNMENT + (ulong)_numel * sizeof(float));
                memset((void*)h_grad,
                    0,
                    ALIGNMENT + sizeof(float) * (ulong)_numel);
            }
            // We need to ensure that the host memory is aligned to 4K
            if (h_data != null)
                data = (float*)(((ulong)h_data + (ALIGNMENT - 1)) & (~(ALIGNMENT - 1)));
            if (h_grad != null)
                grad = (float*)(((ulong)h_grad + (ALIGNMENT - 1)) & (~(ALIGNMENT - 1)));
        }

        void Free() {
            free((void*)Interlocked.Exchange(
                ref h_data,
                IntPtr.Zero));
            free((void*)Interlocked.Exchange(
                ref h_grad,
                IntPtr.Zero));
        }

        public ulong numbytes {
            get {
                ulong numbytes = 0;
                if (h_data != null) {
                    numbytes += ALIGNMENT + (ulong)_numel * sizeof(float);
                }
                if (h_grad != null) {
                    numbytes += ALIGNMENT + (ulong)_numel * sizeof(float);
                }
                return numbytes;
            }
        }

        public uint numel() {
            return _numel;
        }

        public void numel(uint value) {
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