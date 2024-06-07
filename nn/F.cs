namespace nn {
    using System;
    using System.Runtime.ConstrainedExecution;
    using System.Threading.Tasks;

    public static unsafe partial class F {
        public unsafe class Kernel : CriticalFinalizerObject, IDisposable {
            ~Kernel() {
                Dispose(disposing: false);
            }

            public void Dispose() {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing) {
            }
        }

        public static double no_loss(
            Tensor output,
            float[] target) {

            if (output is null) throw new ArgumentNullException(nameof(output));
            if (target is null) throw new ArgumentNullException(nameof(target));

            return 0;
        }

        public static double mse_loss(
            float* _Out,       /* [N] */
            float* d_Out,       /* [N] */
            uint N,
            float* target) {

            if (_Out == null) throw new ArgumentNullException(nameof(_Out));
            if (d_Out == null) throw new ArgumentNullException(nameof(d_Out));

            if (target is null) throw new ArgumentNullException(nameof(target));

            double sum = 0f;

            for (int n = 0; n < N; n++) {
                double diff = (_Out[n] - target[n]);
                sum += diff * diff;
                d_Out[n] = (float)(2.0 * (diff / N));
            }

            return sum / N;
        }

        public static double binary_cross_entropy(
            Tensor logits,
            Tensor target) {

            const double EPSILON = 1e-12f;

            if (logits is null) throw new ArgumentNullException(nameof(logits));
            if (target is null) throw new ArgumentNullException(nameof(target));

            if (target.numel() != logits.numel())
                throw new ArgumentOutOfRangeException(nameof(logits), $"number of '{nameof(logits)}' and '{nameof(target)}' elements must match");

            uint N = logits.numel();

            double acc = 0;

            float* _Out = logits.data;
            float* _Target = target.data;

            for (int n = 0; n < N; n++) {

                if (_Out[n] < 0f || _Out[n] > 1f || float.IsNaN(_Out[n])) throw new ArgumentOutOfRangeException(nameof(logits), $"all elements of '{nameof(logits)}' should be between 0 and 1");
                if (_Target[n] < 0f || _Target[n] > 1f || float.IsNaN(_Target[n])) throw new ArgumentOutOfRangeException(nameof(target), $"all elements of '{nameof(target)}' should be between 0 and 1");

                acc += -(Math.Log(_Out[n] + EPSILON) * _Target[n]
                          + (1.0 - _Target[n]) * Math.Log(1.0 - _Out[n] + EPSILON));

                logits.grad[n] = (float)((_Out[n] - _Target[n]) /
                    Math.Max(
                        (1.0 - _Out[n]) * _Out[n],
                        EPSILON) / N);
            }

            acc /= N;

            return acc;
        }

        public static void dropout_forward_cpu(
            float* _Out,
            float* _In,
            float* _Mask,
            uint N,
            double p,
            bool? training,
            IRNG g) {

            if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p), p, "dropout probability has to be between 0 and 1");

            // during eval dropout layer is pass-through
            if ((training.HasValue
                        && !training.Value) || p == 0 || N == 0) {
                for (int i = 0; i < N; i++) {
                    _Out[i] = _In[i];
                    _Mask[i] = 1f;
                }
                return;
            }

            // 100% droput probability
            if (p == 1) {
                for (int i = 0; i < N; i++) {
                    _Out[i] = 0f;
                    _Mask[i] = 0f;
                }
                return;
            }

            // Inline MCG(1132489760, 2^31 -1) [L'Ecuyer99]
            ulong state_ = g.randint32() % 0x000000007FFFFFFF;
            float scale = p == 1d ? 0f : (1f / (1f - (float)p));
            for (int i = 0; i < N; i++) {
                double sample = (double)(state_ % 0x000000007FFFFFFF) / 0x7FFFFFFF;
                unchecked {
                    state_ = (1132489760 * state_) % 0x000000007FFFFFFF;
                }
                if (sample < 1d - p) {
                    _Mask[i] = 1f;
                    _Out[i] = _In[i] * scale;
                } else {
                    _Mask[i] = 0f;
                    _Out[i] = 0f;
                }
            }
        }

        public static unsafe void dropout_backward_cpu(
            float* _Out,       /* [N] */
            float* d_Out,       /* [N] */
            float* _In,        /* [N] */
            float* d_In,        /* [N] */
            float* _Mask,        /* [N] */
            uint N,
            double p) {

            if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p), p, "dropout probability has to be between 0 and 1");

            float scale = p == 1d ? 0f : (1f / (1f - (float)p));
            for (int n = 0; n < N; n++) {
                d_In[n] = d_Out[n] * _Mask[n] * scale;
            }
        }

        public static unsafe void tanh_forward_cpu(
            float* _Out,       /* [N] */
            float* _In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var x = _In[n];
                float pos = (float)Math.Exp(x), neg = (float)Math.Exp(-x);
                var y = (pos - neg) / (pos + neg);
                _Out[n] = y;
            }
        }

        public static unsafe void tanh_backward_cpu(
            float* _Out,       /* [N] */
            float* d_Out,       /* [N] */
            float* _In,        /* [N] */
            float* d_In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var x = _In[n];
                float pos = (float)Math.Exp(x), neg = (float)Math.Exp(-x);
                var y = (pos - neg) / (pos + neg);
                d_In[n] += (1 - y * y) * d_Out[n];
            }
        }

        public static unsafe void relu_forward_cpu(
            float* _Out,       /* [N] */
            float* _In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var x = _In[n];
                if (x < 0)
                    x = 0;
                _Out[n] = x;
            }
        }

        public static unsafe void relu_backward_cpu(
            float* _Out,       /* [N] */
            float* d_Out,       /* [N] */
            float* _In,        /* [N] */
            float* d_In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var x = _In[n];
                if (x > 0)
                    d_In[n] += d_Out[n];
            }
        }

        public static unsafe void leaky_relu_forward_cpu(
            float* _Out,       /* [N] */
            float* _In,        /* [N] */
            uint N,
            double negval) {

            for (int n = 0; n < N; n++) {
                var x = _In[n];
                if (x < 0)
                    x *= (float)negval;
                _Out[n] = x;
            }
        }

        public static unsafe void leaky_relu_backward_cpu(
            float* _Out,       /* [N] */
            float* d_Out,       /* [N] */
            float* _In,        /* [N] */
            float* d_In,        /* [N] */
            uint N,
            double negval) {

            for (int n = 0; n < N; n++) {
                var x = _In[n];
                d_In[n] += (x > 0) ? d_Out[n] : d_Out[n] * (float)negval;
            }
        }

        public static unsafe void sigmoid_forward_cpu(
            float* _Out,       /* [N] */
            float* _In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var σ = 1.0f
                        / (1.0f + (float)Math.Exp(-_In[n]));

                _Out[n] = σ;
            }
        }

        public static unsafe void sigmoid_backward_cpu(
            float* _Out,       /* [N] */
            float* d_Out,       /* [N] */
            float* _In,        /* [N] */
            float* d_In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var σ = 1.0f
                        / (1.0f + (float)Math.Exp(-_In[n]));

                d_In[n] += σ * (1.0f - σ) * d_Out[n];
            }
        }

        public static unsafe void matmul_forward_naive(
            float* _Out,       /* [B, O] */
            float* _In,        /* [B, I] */
            float* _Weight,    /* [I, O] */
            float* _Bias,      /* [O] */
            uint B,
            uint I,
            uint O,
            int maxDegreeOfParallelism) {

            if (maxDegreeOfParallelism == -1 || maxDegreeOfParallelism > 0) {
                Parallel.For(0, B * O, (bo) => {
                    uint b = (uint)bo / O;
                    uint o = (uint)bo % O;
                    if (b < B && o < O) {
                        float* x = _In + b * I;
                        float* y = _Out + b * O;
                        float acc = _Bias != null ? _Bias[o] : 0;
                        float* w = _Weight + o * I;
                        for (int i = 0; i < I; i++) {
                            acc += w[i] * x[i];
                        }
                        y[o] = (float)acc;
                    }
                });
            } else {
                for (uint bo = 0; bo < B * O; bo++) {
                    uint b = bo / O;
                    uint o = bo % O;
                    if (b < B && o < O) {
                        float* x = _In + b * I;
                        float* y = _Out + b * O;
                        float acc = _Bias != null ? _Bias[o] : 0;
                        float* w = _Weight + o * I;
                        for (int i = 0; i < I; i++) {
                            acc += w[i] * x[i];
                        }
                        y[o] = (float)acc;
                    }
                }
            }
        }

        public static void matmul_backward_naive(
            float* _Out,       /* [B, O] */
            float* _δ_Out,       /* [B, O] */
            float* _In,        /* [B, I] */
            float* _δ_In,        /* [B, I] */
            float* _Weight,    /* [I, O] */
            float* _δ_Weight,    /* [I, O] */
            float* _Bias,      /* [O] */
            float* _δ_Bias,      /* [O] */
            uint B,
            uint I,
            uint O,
            int maxDegreeOfParallelism) {

            if (maxDegreeOfParallelism == -1 || maxDegreeOfParallelism > 0) {
                Parallel.For(0, B, (b) => {
                    for (uint o = 0; o < O; o++) {
                        float* δ_In_b = _δ_In + b * I;
                        float* _Weight_o = _Weight + o * I;
                        float δ = _δ_Out[b * O + o];
                        for (uint i = 0; i < I; i++) {
                            δ_In_b[i] += _Weight_o[i] * δ;
                        }
                    }
                });
                Parallel.For(0, O, (o) => {
                    for (uint b = 0; b < B; b++) {
                        float* _In_b = _In + b * I;
                        float* δ_Weight_o = _δ_Weight + o * I;
                        float δ = _δ_Out[b * O + o];
                        for (uint i = 0; i < I; i++) {
                            δ_Weight_o[i] += _In_b[i] * δ;
                        }
                        if (_δ_Bias != null) {
                            _δ_Bias[o] += δ;
                        }
                    }
                });
            } else {
                for (uint bo = 0; bo < B * O; bo++) {
                    uint b = bo / O;
                    uint o = bo % O;
                    if (b < B && o < O) {
                        float* _In_b = _In + b * I;
                        float* δ_In_b = _δ_In + b * I;
                        float* _Weight_o = _Weight + o * I;
                        float* δ_Weight_o = _δ_Weight + o * I;
                        float δ = _δ_Out[b * O + o];
                        for (uint i = 0; i < I; i++) {
                            δ_In_b[i] += _Weight_o[i] * δ;
                            δ_Weight_o[i] += _In_b[i] * δ;
                        }
                        if (_δ_Bias != null) {
                            _δ_Bias[o] += δ;
                        }
                    }
                }
            }
        }

        public unsafe class MatMul : Kernel {
            public unsafe struct _MatMul {
                public float* _Out;       /* [B, O] */
                public float* d_Out;       /* [B, O] */
                public float* _In;        /* [B, I] */
                public float* d_In;        /* [B, I] */
                public float* _Weight;    /* [I, O] */
                public float* d_Weight;    /* [I, O] */
                public float* _Bias;      /* [O] */
                public float* d_Bias;      /* [O] */
                public uint B;
                public uint I;
                public uint O;
            };

            protected readonly int _maxDegreeOfParallelism;

            public MatMul(int maxDegreeOfParallelism) {
                _maxDegreeOfParallelism = maxDegreeOfParallelism;
            }

            public override string ToString() {
                return $"{GetType().Name}: threads = {_maxDegreeOfParallelism}";
            }

            public virtual void forward(
                float* _Out,       /* [B, O] */
                float* _In,        /* [B, I] */
                float* _Weight,    /* [I, O] */
                float* _Bias,      /* [O] */
                uint B,
                uint I,
                uint O) {

                F.matmul_forward_naive(
                    _Out,
                    _In,
                    _Weight,
                    _Bias,
                    B,
                    I,
                    O,
                    _maxDegreeOfParallelism);
            }

            public virtual void backward(
                float* _Out,       /* [B, O] */
                float* d_Out,       /* [B, O] */
                float* _In,        /* [B, I] */
                float* d_In,        /* [B, I] */
                float* _Weight,    /* [I, O] */
                float* d_Weight,    /* [I, O] */
                float* _Bias,      /* [O] */
                float* d_Bias,      /* [O] */
                uint B,
                uint I,
                uint O) {

                F.matmul_backward_naive(
                    _Out,
                    d_Out,
                    _In,
                    d_In,
                    _Weight,
                    d_Weight,
                    _Bias,
                    d_Bias,
                    B,
                    I,
                    O,
                    _maxDegreeOfParallelism);
            }
        }
    }
}