using System;
using System.Runtime.ConstrainedExecution;

using static cuda;
using static nvrtc;
using static std;

namespace nn.dev {
    public unsafe sealed partial class cuTensor : CriticalFinalizerObject, IDisposable {
        uint _numel;

        public cuTensor(uint numel, bool requires_grad = true) : base() {
            cuMemAlloc(numel, requires_grad, out data, out grad);
        }

        ~cuTensor() {
            Dispose();
        }

        public void Dispose() {
            cuMemFree();
            GC.SuppressFinalize(this);
        }

        public readonly IntPtr data;
        public readonly IntPtr grad;

        void cuMemAlloc(uint numel, bool requires_grad, out IntPtr data, out IntPtr grad) {
            data = IntPtr.Zero; grad = IntPtr.Zero;
            try {
                if (numel > 0) {
                    checkCudaErrors(cuMemAlloc_v2(out data, (ulong)numel * sizeof(float)));
                }
                if (numel > 0 && requires_grad) {
                    checkCudaErrors(cuMemAlloc_v2(out grad, (ulong)numel * sizeof(float)));
                }
                _numel = numel;
            } catch {
                cuMemFree_v2(ref data);
                cuMemFree_v2(ref grad);
                throw;
            }
        }

        void cuMemFree() {
            cuMemFree_v2(data);
            cuMemFree_v2(grad);
        }

        public ulong numbytes {
            get {
                ulong numbytes = 0;
                if (data != IntPtr.Zero) {
                    numbytes += (ulong)_numel * sizeof(float);
                }
                if (grad != IntPtr.Zero) {
                    numbytes += (ulong)_numel * sizeof(float);
                }
                return numbytes;
            }
        }

        public DataType dtype() {
            return DataType.float32;
        }

        public Device device() {
            return Device.cuda;
        }

        public uint numel() {
            return _numel;
        }

        public void resize(uint value) {
            if (_numel >= value) {
                _numel = value;
            } else {
                throw new NotImplementedException();
            }
        }

        public void zero_grad() {
            // kernel32.ZeroMemory(
            //     grad,
            //     (UIntPtr)((ulong)numel() * sizeof(float)));
        }
    }

    public class cuMatMulA : F.MatMul {
        static string CU = @"
             extern ""C"" __global__  void matmul_forward_cu(
                 float* _Out,       /* [B, O] */
                 float* _In,        /* [B, I] */
                 float* _Weight,    /* [I, O] */
                 float* _Bias,      /* [O] */
                 int B,
                 int I,
                 int O) {

                int b = blockIdx.x * blockDim.x + threadIdx.x;

                if (b < B) {
                    float* x = _In + b * I;
                    float* y = _Out + b * O;

                    for (int o = 0; o < O; o++) {
                        float acc = _Bias ? _Bias[o] : 0;
                        float* w = _Weight + o * I;
                        for (int i = 0; i < I; i++) {
                            acc += w[i] * x[i];
                        }
                        y[o] = (float)acc;
                    }
                }

            }";

        IntPtr _matmul_forward_cu;

        public cuMatMulA() : base(-1) {
            printf("> Compiling CUDA kernels...\n");
            byte[] ptx = nvrtcCompileFromSourceCode(CU, "matmul_forward_cu");
            checkCudaErrors(cuModuleLoadData(out var cuModule, ptx));
            checkCudaErrors(cuModuleGetFunction(
                out _matmul_forward_cu,
                cuModule,
                "matmul_forward_cu"));
            printf("> Done.\n\n");
        }

        protected override void Dispose(bool disposing) {
            cu_In?.Dispose();
            cu_Out?.Dispose();
            cu_Weight?.Dispose();
            cu_Bias?.Dispose();
            base.Dispose(disposing);
        }

        cuTensor cu_In;
        cuTensor cu_Out;
        cuTensor cu_Weight;
        cuTensor cu_Bias;

        public override unsafe void forward(
            float* _Out, float* _In, float* _Weight, float* _Bias, uint B, uint I, uint O) {

            if (cu_In == null) {
                cu_In = new cuTensor(B * I, requires_grad: true);
            } else if (cu_In.numel() != B * I) {
                cu_In.resize(B * I);
            }

            if (cu_Out == null) {
                cu_Out = new cuTensor(B * O, requires_grad: true);
            } else if (cu_Out.numel() != B * O) {
                cu_Out.resize(B * O);
            }

            if (cu_Weight == null) {
                cu_Weight = new cuTensor(I * O, requires_grad: true);
            } else if (cu_Weight.numel() != I * O) {
                cu_Weight.resize(I * O);
            }

            if (cu_Bias == null) {
                cu_Bias = new cuTensor(O, requires_grad: true);
            } else if (cu_Bias.numel() != O) {
                cu_Bias.resize(O);
            }

            IntPtr _d_data_In = cu_In.data;
            IntPtr _d_data_Out = cu_Out.data;
            IntPtr _d_data_Weight = cu_Weight.data;
            IntPtr _d_data_Bias = cu_Bias.data;

            checkCudaErrors(cuMemcpyHtoD_v2(_d_data_Out, _Out, cu_Out.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_data_In, _In, cu_In.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_data_Weight, _Weight, cu_Weight.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_data_Bias, _Bias, cu_Bias.numel() * sizeof(float)));

            uint block_size = 32;

            void*[] args = { &_d_data_Out, &_d_data_In, &_d_data_Weight, &_d_data_Bias, &B, &I, &O };

            checkCudaErrors(cuLaunchKernel(
                _matmul_forward_cu,
                CEIL_DIV(B, block_size), 1, 1,
                block_size, 1, 1,
                0,
                IntPtr.Zero,
                args,
                null));

            checkCudaErrors(cuMemcpyDtoH_v2(_Out, _d_data_Out, cu_Out.numel() * sizeof(float)));
            // checkCudaErrors(cuMemcpyDtoH_v2(_In, _d_data_In, cu_In.numel() * sizeof(float)));
            // checkCudaErrors(cuMemcpyDtoH_v2(_Weight, _d_data_Weight, cu_Weight.numel() * sizeof(float)));
            // checkCudaErrors(cuMemcpyDtoH_v2(_Bias, _d_data_Bias, cu_Bias.numel() * sizeof(float)));
        }
    }
}