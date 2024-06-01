using System;
using System.Runtime.ConstrainedExecution;

namespace nn {
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
            float[] target) {

            if (_Out == null) throw new ArgumentNullException(nameof(_Out));
            if (d_Out == null) throw new ArgumentNullException(nameof(d_Out));

            if (target is null) throw new ArgumentNullException(nameof(target));
            
            int N = target.Length;

            double sum = 0f;

            for (int n = 0; n < N; n++) {
                double diff = (_Out[n] - target[n]);
                sum += diff * diff;
                d_Out[n] = (float)(2.0 * (diff / N));
            }

            return sum / N;
        }

        public static double binary_cross_entropy(
            Tensor output,
            float[] target) {

            const double EPSILON = 1e-12f;

            if (output is null) throw new ArgumentNullException(nameof(output));
            if (target is null) throw new ArgumentNullException(nameof(target));

            int N = target.Length;

            double sum = 0;

            for (int n = 0; n < N; n++) {

                if (output.data[n] < 0f || output.data[n] > 1f || float.IsNaN(output.data[n])) throw new ArgumentOutOfRangeException(nameof(target), $"all elements of '{nameof(output)}' should be between 0 and 1");
                if (target[n] < 0f || target[n] > 1f || float.IsNaN(target[n])) throw new ArgumentOutOfRangeException(nameof(target), $"all elements of '{nameof(target)}' should be between 0 and 1");

                sum += -(Math.Log(output.data[n] + EPSILON) * target[n] + (1.0 - target[n]) * Math.Log(1.0 - output.data[n] + EPSILON));

                output.grad[n] = (float)((output.data[n] - target[n]) /
                    Math.Max(
                        (1.0 - output.data[n]) * output.data[n],
                        EPSILON) / N);
            }

            sum /= N;

            return sum;
        }

        public static unsafe void relu_forward_cpu(
            float* _Out,       /* [N] */
            float* _In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var σ = _In[n];
                if (σ <= 0)
                    σ = 0;
                _Out[n] = σ;
            }
        }

        public static unsafe void relu_backward_cpu(
            float* _Out,       /* [N] */
            float* d_Out,       /* [N] */
            float* _In,        /* [N] */
            float* d_In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var σ = _In[n];
                if (σ > 0)
                    d_In[n] += d_Out[n];
            }
        }

        public static unsafe void sigmoid_forward_cpu(
            float* _Out,       /* [N] */
            float* _In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var σ = 1.0f / (1.0f + (float)Math.Exp(-_In[n]));
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
                var σ = 1.0f / (1.0f + (float)Math.Exp(-_In[n]));
                d_In[n] += σ * (1.0f - σ) * d_Out[n];
            }
        }

        public static unsafe void matmul_forward_cpu(
            float* _Out,       /* [B, O] */
            float* _In,        /* [B, I] */
            float* _Weight,    /* [I, O] */
            float* _Bias,      /* [O] */
            uint B,
            uint I,
            uint O) {

            for (int b = 0; b < B; b++) {
                float* x = _In + b * I;
                float* y = _Out + b * O;
                for (int o = 0; o < O; o++) {
                    float acc = _Bias != null ? _Bias[o] : 0;
                    float* w = _Weight + o * I;
                    for (int i = 0; i < I; i++) {
                        acc += w[i] * x[i];
                    }
                    y[o] = (float)acc;
                }
            }
        }

        public static void matmul_backward_cpu(
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

            for (int b = 0; b < B; b++) {
                float* p_In = _In + b * I;
                float* p_d_In = d_In + b * I;
                for (int o = 0; o < O; o++) {
                    float δf = d_Out[b * O + o];
                    float* p_Weight = _Weight + o * I;
                    float* p_d_Weight = d_Weight + o * I;
                    for (int i = 0; i < I; i++) {
                        p_d_In[i] += p_Weight[i] * δf;
                        p_d_Weight[i] += p_In[i] * δf;
                    }
                    if (d_Bias != null) {
                        d_Bias[o] += δf;
                    }
                }
            }
        }
    }
}