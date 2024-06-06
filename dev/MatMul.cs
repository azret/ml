using System;
using nn.CPU;

using static kernel32;
using static std;
using static cuda;

namespace nn.dev {
    static unsafe partial class MatMul_ {
        static unsafe int Main() {
            checkCudaErrors(cuInit());
            checkCudaErrors(cuDeviceGet(out var dev, 0));
            cuPrintDeviceInfo(dev);
            checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, dev));
            checkCudaErrors(cuCtxSetCurrent(ctx));
            cuPrintCurrentContextInfo();

            Console.WriteLine();

            printf("> Compiling CPU & CUDA kernels...\n");

            F.MatMul[] kernels;

            kernels = new []
            {
                new F.MatMul(0),
                new F.MatMul(-1),
                new MatMulAVX2(0),
                new MatMulAVX2(-1),
                new cuMatMulA(32),
                new cuMatMulA(64),
                new cuMatMulA(128),
                new cuMatMulA(256),
                new cuMatMulA(512),
                new cuMatMulA(1024),
            };

            printf("> Done.\n\n");

            uint B = 64;
            uint I = 1024;
            uint O = I * 4;

            Tensor _h_Mem_Out = new Tensor(B * O);
            Tensor _h_Mem_In = new Tensor(B * I);
            Tensor _h_Mem_Weight = new Tensor(O * I);
            Tensor _h_Mem_Bias = new Tensor(O);

            ulong seed = 37;

            for (int i = 0; i < _h_Mem_In.numel(); i++) { _h_Mem_In.data[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _h_Mem_In.numel(); i++) { _h_Mem_In.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _h_Mem_Weight.numel(); i++) { _h_Mem_Weight.data[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _h_Mem_Weight.numel(); i++) { _h_Mem_Weight.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _h_Mem_Bias.numel(); i++) { _h_Mem_Bias.data[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _h_Mem_Bias.numel(); i++) { _h_Mem_Bias.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _h_Mem_Out.numel(); i++) { _h_Mem_Out.data[i] = urandf(&seed) * 2.0f - 1.0f; }
            for (int i = 0; i < _h_Mem_Out.numel(); i++) { _h_Mem_Out.grad[i] = urandf(&seed) * 2.0f - 1.0f; }

            F.matmul_forward_naive(
                _h_Mem_Out.data,
                _h_Mem_In.data,
                _h_Mem_Weight.data,
                _h_Mem_Bias.data,
                B,
                I,
                O,
                0);

            F.matmul_backward_naive(
                _h_Mem_Out.data,
                _h_Mem_Out.grad,
                _h_Mem_In.data,
                _h_Mem_In.grad,
                _h_Mem_Weight.data,
                _h_Mem_Weight.grad,
                _h_Mem_Bias.data,
                _h_Mem_Bias.grad,
                B,
                I,
                O,
                0);

            Tensor _k_Mem_Out = new Tensor(B * O);
            Tensor _k_Mem_In = new Tensor(B * I);
            Tensor _k_Mem_Weight = new Tensor(O * I);
            Tensor _k_Mem_Bias = new Tensor(O);

            bool all_ok = true;

            for (int kernel = 0; kernel < kernels.Length; kernel++) {
                var MatMul = kernels[kernel];

                seed = 37;

                for (int i = 0; i < _k_Mem_In.numel(); i++) { _k_Mem_In.data[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _k_Mem_In.numel(); i++) { _k_Mem_In.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _k_Mem_Weight.numel(); i++) { _k_Mem_Weight.data[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _k_Mem_Weight.numel(); i++) { _k_Mem_Weight.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _k_Mem_Bias.numel(); i++) { _k_Mem_Bias.data[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _k_Mem_Bias.numel(); i++) { _k_Mem_Bias.grad[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _k_Mem_Out.numel(); i++) { _k_Mem_Out.data[i] = urandf(&seed) * 2.0f - 1.0f; }
                for (int i = 0; i < _k_Mem_Out.numel(); i++) { _k_Mem_Out.grad[i] = urandf(&seed) * 2.0f - 1.0f; }

                Console.WriteLine($"== kernel #{kernel} ({kernels[kernel]}) ==");

                MatMul.forward(
                    _k_Mem_Out.data,
                    _k_Mem_In.data,
                    _k_Mem_Weight.data,
                    _k_Mem_Bias.data,
                    B,
                    I,
                    O);

                bool ok = validate_results(_k_Mem_Out.data, _h_Mem_Out.data, _k_Mem_Out.numel(), "\nout");
                ok &= validate_results(_k_Mem_In.data, _h_Mem_In.data, _k_Mem_In.numel(), "\nin");
                ok &= validate_results(_k_Mem_Weight.data, _h_Mem_Weight.data, _h_Mem_Weight.numel(), "\nweight");
                ok &= validate_results(_k_Mem_Bias.data, _h_Mem_Bias.data, _k_Mem_Bias.numel(), "\nbias");

                MatMul.backward(
                    _k_Mem_Out.data,
                    _k_Mem_Out.grad,
                    _k_Mem_In.data,
                    _k_Mem_In.grad,
                    _k_Mem_Weight.data,
                    _k_Mem_Weight.grad,
                    _k_Mem_Bias.data,
                    _k_Mem_Bias.grad,
                    B,
                    I,
                    O);

                ok &= validate_results(_k_Mem_Out.grad, _h_Mem_Out.grad, _k_Mem_Out.numel(), "\nout.grad");
                ok &= validate_results(_k_Mem_In.grad, _h_Mem_In.grad, _k_Mem_In.numel(), "\nin.grad");
                ok &= validate_results(_k_Mem_Weight.grad, _h_Mem_Weight.grad, _k_Mem_Weight.numel(), "\nweight.grad");
                ok &= validate_results(_k_Mem_Bias.grad, _h_Mem_Bias.grad, _k_Mem_Bias.numel(), "\nbias.grad");

                all_ok &= ok;

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

            if (all_ok) {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"ALL OK");
                Console.ResetColor();
            } else {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"FAILED.");
                Console.ResetColor();
            }

            Console.WriteLine();

            for (int kernel = 0; kernel < kernels.Length; kernel++) {
                var MatMul = kernels[kernel];

                ulong start = millis();

                for (int i = 0; i < 64; i++) {
                    MatMul.forward(
                        _k_Mem_Out.data,
                        _k_Mem_In.data,
                        _k_Mem_Weight.data,
                        _k_Mem_Bias.data,
                        B,
                        I,
                        O);
                }

                double elapsed = ((double)millis() - start);

                Console.WriteLine($"kernel #{kernel} forward\t({kernels[kernel]})\t{elapsed:0.00} ms");

                start = millis();

                for (int i = 0; i < 64; i++) {
                    MatMul.backward(
                        _k_Mem_Out.data,
                        _k_Mem_Out.grad,
                        _k_Mem_In.data,
                        _k_Mem_In.grad,
                        _k_Mem_Weight.data,
                        _k_Mem_Weight.grad,
                        _k_Mem_Bias.data,
                        _k_Mem_Bias.grad,
                        B,
                        I,
                        O);
                }

                elapsed = ((double)millis() - start);

                Console.WriteLine($"kernel #{kernel} backward\t({kernels[kernel]})\t{elapsed:0.00} ms");
            }

            checkCudaErrors(cuCtxDestroy_v2(ctx));

            Console.WriteLine();
            printf("Press any to continue...");
            Console.Out.Flush();
            Console.ReadKey();

            return 0;
        }

        public static unsafe bool validate_results(float* d_Mem, float* h_Mem, uint N, 
            string name = null,
            float epsilone = 1e-4f,
            bool bPrint = true) {
            if (!string.IsNullOrWhiteSpace(name) && bPrint) {
                Console.WriteLine($"{name}:");
            }
            bool ok = true;
            int faults = 0;
            int prints = 0;
            for (int i = 0; i < N; ++i) {
                if (Math.Abs(d_Mem[i] - h_Mem[i]) > epsilone) {
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