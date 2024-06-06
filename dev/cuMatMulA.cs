using System;
using System.Runtime.ConstrainedExecution;
using System.Threading;

using static cuda;
using static nvrtc;
using static std;

namespace nn.dev {
    public unsafe sealed partial class cuTensor : CriticalFinalizerObject, IDisposable {
        uint _capacity;
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

        public static void Dispose(ref cuTensor t) {
            var p = Interlocked.Exchange(ref t, null);
            if (p != null) {
                p.Dispose();
            }
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
                _capacity = numel;
            } catch {
                cuMemFree_v2(ref data);
                cuMemFree_v2(ref grad);
                throw;
            }
        }

        void cuMemFree() {
            _numel = 0;
            _capacity = 0;
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
            if (value <= _capacity) {
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
        static string MATMUL_FORWARD_CU = @"
             extern ""C"" __global__  void matmul_forward_cu(
                 float* _Out,       /* [B, O] */
                 float* _In,        /* [B, I] */
                 float* _Weight,    /* [I, O] */
                 float* _Bias,      /* [O] */
                 int B,
                 int I,
                 int O) {

                int bo = blockIdx.x * blockDim.x + threadIdx.x;

                int b = bo / O;
                int o = bo % O;

                if (b < B && o < O) {
                    float* x = _In + b * I;
                    float* y = _Out + b * O;
                    float* w = _Weight + o * I;

                    float acc = _Bias ? _Bias[o] : 0;
                    for (int i = 0; i < I; i++) {
                        acc += w[i] * x[i];
                    }

                    y[o] = (float)acc;
                }

            }";

        static string MATMUL_BACKWARD_CU = @"

        extern ""C"" __global__  void matmul_backward_cu_part_1(
            float* d_In,
            float* d_Out,
            float* _Weight,
            int B,
            int I,
            int O) {

                int bo = blockIdx.x * blockDim.x + threadIdx.x;

                int b = bo / O;
                int o = bo % O;

                if (b < B && o < O) {

                    float* dout_bt = d_Out + b * O;
                    float* dinp_bt = d_In + b * I;

                    //for (int o = 0; o < O; o++) {

                        float* wrow = _Weight + o * I;
                        float d = dout_bt[o];
                        for (int i = 0; i < I; i++) {
                            // dinp_bt[i] += wrow[i] * d;
                            atomicAdd(&dinp_bt[i], wrow[i] * d);
                        }

                    //}

                }

            }

        extern ""C"" __global__  void matmul_backward_cu_part_2(
            float* d_Out,
            float* d_Weight,
            float* d_Bias,
            float* _In,
            int B,
            int I,
            int O) {

                int bo = blockIdx.x * blockDim.x + threadIdx.x;

                int b = bo / O;
                int o = bo % O;

                if (b < B && o < O) {

                    // for (int o = 0; o < O; o++) {

                        float* dout_bt = d_Out + b * O;
                        float* inp_bt = _In + b * I;
                        float* dwrow = d_Weight + o * I;
                        float d = dout_bt[o];
                        if (d_Bias) {
                            // d_Bias[o] += d;
                            atomicAdd(&d_Bias[o], d);
                        }
                        for (int i = 0; i < I; i++) {
                            // dwrow[i] += inp_bt[i] * d;
                            atomicAdd(&dwrow[i], inp_bt[i] * d);
                        }

                    // }

                }

            }
    ";

        IntPtr _matmul_forward_cu;
        IntPtr _matmul_backward_cu_part_1;
        IntPtr _matmul_backward_cu_part_2;

        uint block_size;

        public override string ToString() {
            return $"cuMatMulA: block_size = {block_size}";
        }

        public cuMatMulA(uint block_size) : base(-1) {
            switch (block_size) {
                case 32:
                case 64:
                case 128:
                case 256:
                case 512:
                case 1024:
                    this.block_size = block_size;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(block_size));
            }

            byte[] ptx = nvrtcCompileFromSourceCode(MATMUL_FORWARD_CU, "MATMUL_FORWARD_CU");
            checkCudaErrors(cuModuleLoadData(out var cuModule, ptx));

            checkCudaErrors(cuModuleGetFunction(
                out _matmul_forward_cu,
                cuModule,
                "matmul_forward_cu"));

            ptx = nvrtcCompileFromSourceCode(MATMUL_BACKWARD_CU, "MATMUL_BACKWARD_CU");
            checkCudaErrors(cuModuleLoadData(out cuModule, ptx));

            checkCudaErrors(cuModuleGetFunction(
                out _matmul_backward_cu_part_1,
                cuModule,
                "matmul_backward_cu_part_1"));

            checkCudaErrors(cuModuleGetFunction(
                out _matmul_backward_cu_part_2,
                cuModule,
                "matmul_backward_cu_part_2"));
        }

        protected override void Dispose(bool disposing) {
            cuTensor.Dispose(ref _cuIn);
            cuTensor.Dispose(ref _cuOut);
            cuTensor.Dispose(ref _cuWeight);
            cuTensor.Dispose(ref _cuBias);
            base.Dispose(disposing);
        }

        cuTensor _cuIn;
        cuTensor _cuOut;
        cuTensor _cuWeight;
        cuTensor _cuBias;

        public override unsafe void backward(
            float* _Out,
            float* d_Out,
            float* _In,
            float* d_In,
            float* _Weight,
            float* d_Weight,
            float* _Bias,
            float* d_Bias,
            uint B,
            uint I,
            uint O) {

            // =======================

            IntPtr _d_Mem_cu_d_In = _cuIn.grad;
            IntPtr _d_Mem_cu_d_Out = _cuOut.grad;
            IntPtr _d_Mem_cuWeight = _cuWeight.data;

            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cu_d_Out, d_Out, _cuOut.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cu_d_In, d_In, _cuIn.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cuWeight, _Weight, _cuWeight.numel() * sizeof(float)));

            void*[] args1 = { &_d_Mem_cu_d_In, &_d_Mem_cu_d_Out, &_d_Mem_cuWeight, &B, &I, &O };

            // for (int b = 0; b < B; b++) {
            //     float* dout_bt = d_Out + b * O;
            //     float* dinp_bt = d_In + b * I;
            // 
            //     for (int o = 0; o < O; o++) {
            //         float* wrow = _Weight + o * I;
            //         float d = dout_bt[o];
            //         for (int i = 0; i < I; i++) {
            //             dinp_bt[i] += wrow[i] * d;
            //         }
            //     }
            // }

            checkCudaErrors(cuLaunchKernel(
                _matmul_backward_cu_part_1,
                CEIL_DIV(B * O, block_size), 1, 1,
                block_size, 1, 1,
                0,
                IntPtr.Zero,
                args1,
                null));

            checkCudaErrors(cuMemcpyDtoH_v2(d_In, _d_Mem_cu_d_In, _cuIn.numel() * sizeof(float)));

            // =======================

            // for (int b = 0; b < B; b++) {
            //     for (int o = 0; o < O; o++) {
            //         float* dout_bt = d_Out + b * O;
            //         float* inp_bt = _In + b * I;
            //         float* dwrow = d_Weight + o * I;
            //         float d = dout_bt[o];
            //         if (d_Bias != null) { d_Bias[o] += d; }
            //         for (int i = 0; i < I; i++) {
            //             dwrow[i] += inp_bt[i] * d;
            //         }
            //     }
            // }

            IntPtr _d_Mem_cu_In = _cuIn.data;
            IntPtr _d_Mem_cu_d_Weight = _cuWeight.grad;
            IntPtr _d_Mem_cu_d_Bias = _cuBias.grad;

            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cu_In, _In, _cuIn.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cu_d_Weight, d_Weight, _cuWeight.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cu_d_Bias, d_Bias, _cuBias.numel() * sizeof(float)));

            void*[] args2 = { &_d_Mem_cu_d_Out, &_d_Mem_cu_d_Weight, &_d_Mem_cu_d_Bias, &_d_Mem_cu_In, &B, &I, &O };

            checkCudaErrors(cuLaunchKernel(
                _matmul_backward_cu_part_2,
                CEIL_DIV(B * O, block_size), 1, 1,
                block_size, 1, 1,
                0,
                IntPtr.Zero,
                args2,
                null));

            checkCudaErrors(cuMemcpyDtoH_v2(d_Weight, _d_Mem_cu_d_Weight, _cuWeight.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyDtoH_v2(d_Bias, _d_Mem_cu_d_Bias, _cuBias.numel() * sizeof(float)));

        }

        public override unsafe void forward(
            float* _Out, float* _In, float* _Weight, float* _Bias, uint B, uint I, uint O) {

            if (_cuIn == null) {
                _cuIn = new cuTensor(B * I, requires_grad: true);
            } else if (_cuIn.numel() != B * I) {
                _cuIn.resize(B * I);
            }

            if (_cuOut == null) {
                _cuOut = new cuTensor(B * O, requires_grad: true);
            } else if (_cuOut.numel() != B * O) {
                _cuOut.resize(B * O);
            }

            if (_cuWeight == null) {
                _cuWeight = new cuTensor(I * O, requires_grad: true);
            } else if (_cuWeight.numel() != I * O) {
                _cuWeight.resize(I * O);
            }

            if (_cuBias == null) {
                _cuBias = new cuTensor(O, requires_grad: true);
            } else if (_cuBias.numel() != O) {
                _cuBias.resize(O);
            }

            IntPtr _d_Mem_cuIn = _cuIn.data;
            IntPtr _d_Mem_cuOut = _cuOut.data;
            IntPtr _d_Mem_cuWeight = _cuWeight.data;
            IntPtr _d_Mem_cuBias = _cuBias.data;

            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cuOut, _Out, _cuOut.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cuIn, _In, _cuIn.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cuWeight, _Weight, _cuWeight.numel() * sizeof(float)));
            checkCudaErrors(cuMemcpyHtoD_v2(_d_Mem_cuBias, _Bias, _cuBias.numel() * sizeof(float)));

            void*[] args = { &_d_Mem_cuOut, &_d_Mem_cuIn, &_d_Mem_cuWeight, &_d_Mem_cuBias, &B, &I, &O };

            checkCudaErrors(cuLaunchKernel(
                _matmul_forward_cu,
                CEIL_DIV(B * O, block_size), 1, 1,
                block_size, 1, 1,
                0,
                IntPtr.Zero,
                args,
                null));

            checkCudaErrors(cuMemcpyDtoH_v2(_Out, _d_Mem_cuOut, _cuOut.numel() * sizeof(float)));

            // checkCudaErrors(cuMemcpyDtoH_v2(_In, _d_data_In, cu_In.numel() * sizeof(float)));
            // checkCudaErrors(cuMemcpyDtoH_v2(_Weight, _d_data_Weight, cu_Weight.numel() * sizeof(float)));
            // checkCudaErrors(cuMemcpyDtoH_v2(_Bias, _d_data_Bias, cu_Bias.numel() * sizeof(float)));
        }
    }
}