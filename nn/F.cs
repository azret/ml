using System;
using System.ComponentModel;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using System.Threading;

using static kernel32;

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

        /// <summary>
        /// Base class for all MatMul kernels
        /// </summary>
        public unsafe abstract class MatMul : Kernel {
            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public unsafe delegate void MatMul_Forward(
                float* _Out,       /* [B, O] */
                float* _In,        /* [B, I] */
                float* _Weight,    /* [I, O] */
                float* _Bias,      /* [O] */
                uint B,
                uint I,
                uint O);

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public unsafe delegate void MatMul_Backward(
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
                uint O);

            public abstract void forward(
                float* _Out,       /* [B, O] */
                float* _In,        /* [B, I] */
                float* _Weight,    /* [I, O] */
                float* _Bias,      /* [O] */
                uint B,
                uint I,
                uint O);

            public abstract void backward(
                Tensor _Out,       /* [B, O] */
                Tensor _In,        /* [B, I] */
                Tensor _Weight,    /* [I, O] */
                Tensor _Bias,      /* [O] */
                uint B,
                uint I,
                uint O);
        }

        /// <summary>
        /// A no-op implementation
        /// </summary>
        public unsafe class MatMulN : MatMul {
            public override unsafe void forward(
                float* _Out,
                float* _In,
                float* _Weight,
                float* _Bias,
                uint B,
                uint I,
                uint O) {
            }

            public override void backward(
                Tensor _Out,       /* [B, O] */
                Tensor _In,        /* [B, I] */
                Tensor _Weight,    /* [I, O] */
                Tensor _Bias,      /* [O] */
                uint B,
                uint I,
                uint O) {
            }
        }

        /// <summary>
        ///  A naive reference C# implementation
        /// </summary>
        public unsafe class MatMulA : MatMul {
            public override void forward(
                float* _Out,       /* [B, O] */
                float* _In,        /* [B, I] */
                float* _Weight,    /* [I, O] */
                float* _Bias,      /* [O] */
                uint B,
                uint I,
                uint O) {

                matmul_forward_cpu(
                    _Out,
                    _In,
                    _Weight,
                    _Bias,
                    B,
                    I,
                    O);
            }

            public override void backward(
                Tensor _Out,       /* [B, O] */
                Tensor _In,        /* [B, I] */
                Tensor _Weight,    /* [I, O] */
                Tensor _Bias,      /* [O] */
                uint B,
                uint I,
                uint O) {

                matmul_backward_cpu(
                    _Out,
                    _In,
                    _Weight,
                    _Bias,
                    B,
                    I,
                    O);
            }
        }

        /// <summary>
        /// Base class for all MatMul kernels written in raw assembly
        /// </summary>
        public abstract class MatMulC : MatMulA, IDisposable {
            IntPtr _p_matmul_forward_cpu_ptr;
            IntPtr _p_matmul_backward_cpu_ptr;

            MatMul_Forward _matmul_forward_cpu_func;
            MatMul_Backward _matmul_backward_cpu_func;

            public MatMulC(byte[] matmul_forward_cpu_asm, byte[] matmul_backward_cpu_asm = null) : base() {
                if (!(matmul_forward_cpu_asm is null)) {
                    _matmul_forward_cpu_func = alloc<MatMul_Forward>(matmul_forward_cpu_asm, out _p_matmul_forward_cpu_ptr);
                }
                if (!(matmul_backward_cpu_asm is null)) {
                    _matmul_backward_cpu_func = alloc<MatMul_Backward>(matmul_backward_cpu_asm, out _p_matmul_backward_cpu_ptr);
                }
            }

            protected override void Dispose(bool disposing) {
                free(disposing, ref _p_matmul_backward_cpu_ptr, ref _matmul_forward_cpu_func);
                free(disposing, ref _p_matmul_forward_cpu_ptr, ref _matmul_backward_cpu_func);
            }

            static private TDelegate alloc<TDelegate>(byte[] asm, out IntPtr _func_ptr) {
                _func_ptr = VirtualAlloc(IntPtr.Zero, new IntPtr(asm.Length),
                    AllocationTypes.Commit | AllocationTypes.Reserve, MemoryProtections.ExecuteReadWrite);
                if (_func_ptr == IntPtr.Zero) {
                    throw new OutOfMemoryException();
                }
                Marshal.Copy(asm, 0, _func_ptr, asm.Length);
                return Marshal.GetDelegateForFunctionPointer<TDelegate>(_func_ptr);
            }

            static void free<TDelegate>(bool disposing, ref IntPtr _func_ptr, ref TDelegate func) where TDelegate : class {
                Interlocked.Exchange(ref func, null);
                IntPtr p = Interlocked.Exchange(ref _func_ptr, IntPtr.Zero);
                if (p != IntPtr.Zero) {
                    bool success = VirtualFree(p, IntPtr.Zero, FreeTypes.Release);
                    if (!success && disposing) {
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    }
                }
            }

            public override unsafe void forward(
                float* _Out,       /* [B, O] */
                float* _In,        /* [B, I] */
                float* _Weight,    /* [I, O] */
                float* _Bias,      /* [O] */
                uint B,
                uint I,
                uint O) {

                if (_matmul_forward_cpu_func is null) {
                    F.matmul_forward_cpu(
                        _Out,
                        _In,
                        _Weight,
                        _Bias,
                        B,
                        I,
                        O);
                } else {
                    _matmul_forward_cpu_func(
                        _Out,
                        _In,
                        _Weight,
                        _Bias,
                        B,
                        I,
                        O);
                }
            }

            public override void backward(
                Tensor _Out,       /* [B, O] */
                Tensor _In,        /* [B, I] */
                Tensor _Weight,    /* [I, O] */
                Tensor _Bias,      /* [O] */
                uint B,
                uint I,
                uint O) {

                if (_matmul_backward_cpu_func is null) {
                    F.matmul_backward_cpu(
                        _Out,
                        _In,
                        _Weight,
                        _Bias,
                        B,
                        I,
                        O);
                } else {
                    _matmul_backward_cpu_func(
                        _Out.data,
                        _Out.grad,
                        _In.data,
                        _In.grad,
                        _Weight.data,
                        _Weight.grad,
                        _Bias.data,
                        _Bias.grad,
                        B,
                        I,
                        O);
                }
            }
        }
    }
}