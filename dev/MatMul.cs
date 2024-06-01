using System;
using nn.CPU;
using static kernel32;
using static std;

namespace nn.dev {
    static unsafe partial class MatMul_ {

        static F.MatMul[] kernels;

        static unsafe int Main() {
            // checkCudaErrors(cuInit());
            // checkCudaErrors(cuDeviceGet(out var dev, 0));
            // cuPrintDeviceInfo(dev);
            // checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, dev));
            // checkCudaErrors(cuCtxSetCurrent(ctx));
            // cuPrintCurrentContextInfo();

            Console.WriteLine();

            kernels = new F.MatMul[]
            {
                new F.MatMul(),
                new MatMulC(),
                new MatMulAVX(),
                new MatMulAVX2(),
            };

            uint B = 32;
            uint I = 1024;
            uint O = I * 4;

            Tensor _Out = new Tensor(B * O);
            Tensor _In = new Tensor(B * I);
            Tensor _Weight = new Tensor(O * I);
            Tensor _Bias = new Tensor(O);

            ulong seed = 37;

            for (int i = 0; i < _In.numel(); i++) { _In.data[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _In.numel(); i++) { _In.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _Weight.numel(); i++) { _Weight.data[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _Weight.numel(); i++) { _Weight.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _Bias.numel(); i++) { _Bias.data[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _Bias.numel(); i++) { _Bias.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _Out.numel(); i++) { _Out.data[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _Out.numel(); i++) { _Out.grad[i] = urandf(&seed) * 2.0f - 1.0f; }

            F.matmul_forward_cpu(
                _Out.data,
                _In.data,
                _Weight.data,
                _Bias.data,
                B,
                I,
                O);

            F.matmul_backward_cpu(
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

            // checkCudaErrors(cuMemAlloc_v2(out var d_Out_Tmp, (ulong)_Out_Tmp.numel() * sizeof(float)));
            // checkCudaErrors(cuMemAlloc_v2(out var d_In, (ulong)_In.numel() * sizeof(float)));
            // checkCudaErrors(cuMemAlloc_v2(out var d_Weight, (ulong)_Weight.numel() * sizeof(float)));
            // checkCudaErrors(cuMemAlloc_v2(out var d_Bias, (ulong)_Bias.numel() * sizeof(float)));
            // 
            // checkCudaErrors(cuMemcpyHtoD_v2(d_In, _In.data, (ulong)_In.numel() * sizeof(float)));
            // checkCudaErrors(cuMemcpyHtoD_v2(d_Weight, _Weight.data, (ulong)_Weight.numel() * sizeof(float)));
            // checkCudaErrors(cuMemcpyHtoD_v2(d_Bias, _Bias.data, (ulong)_Bias.numel() * sizeof(float)));

            Tensor _Out_Tmp = new Tensor(B * O);
            Tensor _In_Tmp = new Tensor(B * I);
            Tensor _Weight_Tmp = new Tensor(O * I);
            Tensor _Bias_Tmp = new Tensor(O);

            for (int kernel = 0; kernel < kernels.Length; kernel++) {
                var MatMul = kernels[kernel];

                seed = 37;

                for (int i = 0; i < _In_Tmp.numel(); i++) { _In_Tmp.data[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _In_Tmp.numel(); i++) { _In_Tmp.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _Weight_Tmp.numel(); i++) { _Weight_Tmp.data[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _Weight_Tmp.numel(); i++) { _Weight_Tmp.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _Bias_Tmp.numel(); i++) { _Bias_Tmp.data[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _Bias_Tmp.numel(); i++) { _Bias_Tmp.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _Out_Tmp.numel(); i++) { _Out_Tmp.data[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _Out_Tmp.numel(); i++) { _Out_Tmp.grad[i] = urandf(&seed) * 2.0f - 1.0f; }

                MatMul.forward(
                    _Out_Tmp.data,
                    _In_Tmp.data,
                    _Weight_Tmp.data,
                    _Bias_Tmp.data,
                    B,
                    I,
                    O);

                MatMul.backward(
                    _Out_Tmp.data,
                    _Out_Tmp.grad,
                    _In_Tmp.data,
                    _In_Tmp.grad,
                    _Weight_Tmp.data,
                    _Weight_Tmp.grad,
                    _Bias_Tmp.data,
                    _Bias_Tmp.grad,
                    B,
                    I,
                    O);

                Console.WriteLine($"== kernel #{kernel} ({kernels[kernel].GetType()}) ==");

                bool ok = validate_results(_Out_Tmp.data, _Out.data, _Out_Tmp.numel(), "\nout");
                ok &= validate_results(_In_Tmp.grad, _In.grad, _In_Tmp.numel(), "\nd_in");
                ok &= validate_results(_Weight_Tmp.grad, _Weight.grad, _Weight_Tmp.numel(), "\nd_weight");
                ok &= validate_results(_Bias_Tmp.grad, _Bias.grad, _Bias_Tmp.numel(), "\nd_bias");

                Console.WriteLine();
                if (ok) {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"OK");
                    Console.ResetColor();
                } else {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"FAILED.");
                    Console.ResetColor();
                }

                Console.WriteLine();
            }


            for (int kernel = 0; kernel < kernels.Length; kernel++) {
                var MatMul = kernels[kernel];

                ulong start = millis();
                for (int i = 0; i < 64; i++) {
                    MatMul.forward(
                        _Out_Tmp.data,
                        _In_Tmp.data,
                        _Weight_Tmp.data,
                        _Bias_Tmp.data,
                        B,
                        I,
                        O);
                }
                double elapsed = ((double)millis() - start);
                Console.WriteLine($"kernel #{kernel} forward ({kernels[kernel].GetType()}), {elapsed:0.00} ms");

                start = millis();
                for (int i = 0; i < 64; i++) {
                    MatMul.backward(
                        _Out_Tmp.data,
                        _Out_Tmp.grad,
                        _In_Tmp.data,
                        _In_Tmp.grad,
                        _Weight_Tmp.data,
                        _Weight_Tmp.grad,
                        _Bias_Tmp.data,
                        _Bias_Tmp.grad,
                        B,
                        I,
                        O);
                }
                
                elapsed = ((double)millis() - start);
                Console.WriteLine($"kernel #{kernel} backward ({kernels[kernel].GetType()}), {elapsed:0.00} ms");
            }

            // checkCudaErrors(cuCtxDestroy_v2(ctx));

            Console.WriteLine();
            printf("Press any to continue...");
            Console.Out.Flush();
            Console.ReadKey();

            return 0;
        }

        public static unsafe bool validate_results(float* d_Mem, float* h_Mem, uint N, string name = null, bool bPrint = true) {
            if (!string.IsNullOrWhiteSpace(name) && bPrint) {
                Console.WriteLine($"{name}:");
            }
            bool ok = true;
            int faults = 0;
            int prints = 0;
            for (int i = 0; i < N; ++i) {
                if (Math.Abs(d_Mem[i] - h_Mem[i]) > 1e-4f) {
                    ok = false;
                    if (faults < 7 && bPrint) {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine($"ERROR: CPU: {h_Mem[i]} != GPU: {d_Mem[i]}");
                        Console.ResetColor();
                    }
                    faults++;
                    break;
                } else {
                    if (faults == 0 && prints < 5 && bPrint) Console.WriteLine($"OK: CPU: {h_Mem[i]} == GPU: {d_Mem[i]}");
                    prints++;
                }
            }
            return ok;
        }
    }
}