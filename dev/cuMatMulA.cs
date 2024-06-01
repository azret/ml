using System;

using static cuda;
using static nvrtc;
using static std;

namespace nn.dev {
    static unsafe partial class MatMul_ {
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
 
     int bo = blockIdx.x * blockDim.x + threadIdx.x;
 
     int b = bo / O;
     int o = bo % O;
 
     if (b < B && o < O) {
     
        _Out[b * O + o] = _Bias ? _Bias[o] : 0;
     
        for (int i = 0; i < I; i++) {
            _Out[b * O + o] += _Weight[o * I + i] * _In[b * I + i];
        }
     
     }
 
 }
 
 ";
            IntPtr _matmul_forward_cu;

            public cuMatMulA() {
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
                base.Dispose(disposing);
            }

            public override unsafe void forward(float* _Out, float* _In, float* _Weight, float* _Bias, uint B, uint I, uint O) {

                uint block_size = 1024;

                void*[] args = { &_Out, &_In, &_Weight, &_Bias, &B, &I, &O };

                checkCudaErrors(cuLaunchKernel(
                    _matmul_forward_cu,
                    CEIL_DIV((uint)(B * O), block_size), 1, 1,
                    block_size, 1, 1,
                    0,
                    IntPtr.Zero,
                    args,
                    null));

            }
        }
    }
}