using System;

namespace nn {
    public static unsafe partial class F {
        public static double no_loss(
            Tensor output,
            float[] target) {

            if (output is null) throw new ArgumentNullException(nameof(output));
            if (target is null) throw new ArgumentNullException(nameof(target));

            return 0;
        }

        public static double mse_loss(
            Tensor output,
            float[] target) {

            if (output is null) throw new ArgumentNullException(nameof(output));
            if (target is null) throw new ArgumentNullException(nameof(target));
            
            int N = target.Length;

            double sum = 0f;

            for (int n = 0; n < N; n++) {
                double diff = (output.data[n] - target[n]);
                sum += diff * diff;
                output.grad[n] = -(float)(2.0 * (diff / N));
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

                if (output.data[n] < 0f || output.data[n] > 1f) throw new ArgumentOutOfRangeException(nameof(target), $"all elements of '{nameof(output)}' should be between 0 and 1");
                if (target[n] < 0f || target[n] > 1f) throw new ArgumentOutOfRangeException(nameof(target), $"all elements of '{nameof(target)}' should be between 0 and 1");

                sum += -(Math.Log(output.data[n] + EPSILON) * target[n] + (1.0 - target[n]) * Math.Log(1.0 - output.data[n] + EPSILON));

                output.grad[n] = -(float)((output.data[n] - target[n]) /
                    Math.Max(
                        (1.0 - output.data[n]) * output.data[n],
                        EPSILON) / N);
            }

            sum /= N;

            return sum;
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
            Tensor _Out,       /* [N] */
            Tensor _In,        /* [N] */
            uint N) {

            for (int n = 0; n < N; n++) {
                var σ = 1.0f / (1.0f + (float)Math.Exp(-_In.data[n]));
                _In.grad[n] += σ * (1.0f - σ) * _Out.grad[n];
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
                float* p_In_bt = _In + b * I;
                float* p_Out_bt = _Out + b * O;
                for (int o = 0; o < O; o++) {
                    double sum = _Bias == null ? 0 : _Bias[o];
                    for (int i = 0; i < I; i++) {
                        sum += _Weight[o * I + i] * p_In_bt[i];
                    }
                    p_Out_bt[o] = (float)sum;
                }
            }
        }

        public static void matmul_backward_cpu(
            Tensor _Out,       /* [B, O] */
            Tensor _In,        /* [B, I] */
            Tensor _Weight,    /* [I, O] */
            Tensor _Bias,      /* [O] */
            uint B,
            uint I,
            uint O) {

            for (int b = 0; b < B; b++) {
                for (int o = 0; o < O; o++) {
                    float δf = _Out.grad[b * O + o];
                    for (int i = 0; i < I; i++) {
                        _In.grad[b * I + i] += _Weight.data[o * I + i] * δf;
                    }
                }
            }

            for (int b = 0; b < B; b++) {
                for (int o = 0; o < O; o++) {
                    float δf = _Out.grad[b * O + o];
                    for (int i = 0; i < I; i++) {
                        _Weight.grad[o * I + i] += _In.data[b * I + i] * δf;
                    }
                    if (_Bias != null) {
                        _Bias.grad[o] += δf;
                    }
                }
            }
        }
    }
}