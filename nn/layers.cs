namespace nn {
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;

    public interface ICompute {
        Tensor forward(Tensor input);
        Tensor backward(Tensor output);
    }

    public interface ILayer : ICompute, IDisposable {
        IEnumerable<Tensor> parameters();
        void eval();
        void train();
    }

    public sealed class Identity : ILayer {
        public void Dispose() {
        }

        public IEnumerable<Tensor> parameters() { yield break; }
        public void eval() { }
        public void train() { }

        public Tensor forward(Tensor input) {
            return input;
        }

        public Tensor backward(Tensor output) {
            return output;
        }
    }

    public unsafe sealed class Sequential : ILayer, IEnumerable<ILayer> {
        ILayer[] Block = Array.Empty<ILayer>();

        public IEnumerator<ILayer> GetEnumerator() {
            return ((IEnumerable<ILayer>)Block).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator() {
            return Block.GetEnumerator();
        }

        public Sequential(params ILayer[] modules) {
            foreach (var m in modules) {
                add(m);
            }
        }

        public void add(ILayer m) {
            var M = new ILayer[Block.Length + 1];
            Array.Copy(Block, M, Block.Length);
            M[Block.Length] = m;
            Block = M;
        }

        public void Dispose() {
            var M = Block;
            Block = null;
            for (int m = M.Length - 1; m >= 0; m--) {
                M[m].Dispose();
            }
        }

        public IEnumerable<Tensor> parameters() {
            foreach (ILayer m in Block) {
                foreach (var p in m.parameters()) {
                    yield return p;
                }
            }
        }

        public void eval() {
            foreach (ILayer m in Block) {
                m.eval();
            }
        }

        public void train() {
            foreach (ILayer m in Block) {
                m.train();
            }
        }

        public Tensor forward(Tensor input) {
            for (int m = 0; m < Block.Length; m++) {
                input = Block[m].forward(input);
            }
            return input;
        }

        public Tensor backward(Tensor output) {
            for (int m = Block.Length - 1; m >= 0; m--) {
                output = Block[m].backward(output);
            }
            return output;
        }
    }

    public unsafe class ReLU : ILayer {
        Tensor _In, _Out;

        public void Dispose() {
            if (_Out != null) _Out.Dispose();
            _Out = null;
            if (_In != null) _In.Dispose();
            _In = null;
        }

        public IEnumerable<Tensor> parameters() { yield break; }

        public void eval() { }

        public void train() { }

        protected virtual void forward(
            float* _Out,       /* [N] */
            float* _In,        /* [N] */
            uint N) {

            F.relu_forward_cpu(
                _Out,
                _In,
                N);
        }

        public Tensor forward(Tensor input) {
            uint N = input.numel();

            // Dynamically create space for input and output to accommodate the batch size

            if (_In is null) {
                _In = new Tensor(N, requires_grad: true);
            } else {
                if (_In.numel() != N) {
                    _In.resize((N));
                }
            }
            if (_Out is null) {
                _Out = new Tensor(N, requires_grad: true);
            } else {
                if (_Out.numel() != N) {
                    _Out.resize((N));
                }
            }

            _In.fill_(input.data, N);

            forward(
                _Out.data,
                _In.data,
                N);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            uint N = _In.numel();

            if (output.numel() != N) throw new InvalidOperationException();

            _In.zero_grad();

            F.relu_backward_cpu(
                output.data,
                output.grad,
                _In.data,
                _In.grad,
                N);

            return _In;
        }
    }

    public unsafe class Dropout : ILayer {
        IRNG g;

        public readonly double p;

        Tensor _In;
        Tensor _Out;
        Tensor _Mask;

        bool? training = null;

        public Dropout(IRNG g, double p = 0.5) {
            this.p = p;
            this.g = g;
        }

        public void Dispose() {
            if (_Out != null) _Out.Dispose();
            _Out = null;
            if (_In != null) _In.Dispose();
            _In = null;
            if (_Mask != null) _Mask.Dispose();
            _Mask = null;
        }

        public IEnumerable<Tensor> parameters() { yield break; }

        public void train() {
            training = true;
        }

        public void eval() {
            training = false;
        }

        public Tensor forward(Tensor input) {
            uint N = input.numel();

            if (_In is null) {
                _In = new Tensor(N, requires_grad: true);
            } else {
                if (_In.numel() != N) {
                    _In.resize((N));
                }
            }
            if (_Out is null) {
                _Out = new Tensor(N, requires_grad: true);
            } else {
                if (_Out.numel() != N) {
                    _Out.resize((N));
                }
            }
            if (_Mask is null) {
                _Mask = new Tensor(N, requires_grad: false);
            } else {
                if (_Mask.numel() != N) {
                    _Mask.resize((N));
                }
            }

            _In.fill_(input.data, N);

            F.dropout_forward_cpu(
                _Out.data,
                _In.data,
                _Mask.data,
                N,
                p,
                training,
                g);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            uint N = _In.numel();

            if (output.numel() != N) throw new InvalidOperationException();

            _In.zero_grad();

            F.dropout_backward_cpu(
                output.data,
                output.grad,
                _In.data,
                _In.grad,
                 _Mask.data,
                N,
                p);

            return _In;
        }
    }

    public unsafe class Sigmoid : ILayer {
        Tensor _In, _Out;

        public void Dispose() {
            if (_Out != null) _Out.Dispose();
            _Out = null;
            if (_In != null) _In.Dispose();
            _In = null;
        }

        public IEnumerable<Tensor> parameters() { yield break; }
        public void eval() { }
        public void train() { }

        protected virtual void forward(
            float* _Out,       /* [N] */
            float* _In,        /* [N] */
            uint N) {

            F.sigmoid_forward_cpu(
                _Out,
                _In,
                N);
        }

        public Tensor forward(Tensor input) {
            uint N = input.numel();

            // Dynamically create space for input and output to accommodate the batch size

            if (_In is null) {
                _In = new Tensor(N, requires_grad: true);
            } else {
                if (_In.numel() != N) {
                    _In.resize((N));
                }
            }
            if (_Out is null) {
                _Out = new Tensor(N, requires_grad: true);
            } else {
                if (_Out.numel() != N) {
                    _Out.resize((N));
                }
            }

            _In.fill_(input.data, N);

            forward(
                _Out.data,
                _In.data,
                N);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            uint N = _In.numel();

            if (output.numel() != N) throw new InvalidOperationException();

            _In.zero_grad();

            F.sigmoid_backward_cpu(
                output.data,
                output.grad,
                _In.data,
                _In.grad,
                N);

            return _In;
        }
    }

    [DebuggerDisplay("nn.Linear ({I}, {O})")]
    public unsafe class Linear : ILayer {
        F.MatMul _MatMul;

        public readonly uint I;
        public readonly uint O;

        Tensor _In;     /* [B, I] */
        Tensor _Out;    /* [B, O] */

        public readonly Tensor _Weight; /* [I, O] */
        public readonly Tensor _Bias;   /* [O] */

        public Linear(int in_features, int out_features, bool bias = true) {
            if (in_features <= 0 || in_features >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(in_features));
            }
            if (out_features <= 0 || out_features >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(out_features));
            }
            I = (uint)in_features;
            O = (uint)out_features;
            _Weight = new Tensor(O * I, requires_grad: true);
            _Bias = bias
                ? new Tensor(O, requires_grad: true)
                : null;
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported) {
                _MatMul = new nn.CPU.MatMulAVX2();
            }
            else if (System.Runtime.Intrinsics.X86.Avx.IsSupported) {
                _MatMul = new nn.CPU.MatMulAVX();
            }
            else {
                _MatMul = new nn.CPU.MatMulC();
            }
        }

        public void Dispose() {
            if (_Out != null) _Out.Dispose();
            _Out = null;
            if (_In != null) _In.Dispose();
            _In = null;
            if (_Bias != null) _Bias.Dispose();
            if (_Weight != null) _Weight.Dispose();
            if (_MatMul != null) _MatMul.Dispose();
            _MatMul = null;
        }

        public IEnumerable<Tensor> parameters() {
            if (_Weight != null) yield return _Weight;
            if (_Bias != null) yield return _Bias;
        }

        public void eval() { }
        public void train() { }

        public Tensor forward(Tensor input) {
            if (_Weight == null) throw new ObjectDisposedException(GetType().FullName);

            uint N = input.numel();

            // Dynamically create space for input and output to accommodate the batch size

            uint B = (N + I - 1) / I;

            if (B * I != N) throw new ArgumentOutOfRangeException(nameof(input));

            if (_In is null) {
                _In = new Tensor(N, requires_grad: true);
            } else {
                if (_In.numel() != B * I) {
                    _In.resize(B * I);
                }
            }

            if (_Out is null) {
                _Out = new Tensor(B * O, requires_grad: true);
            } else {
                if (_Out.numel() != B * O) {
                    _Out.resize(B * O);
                }
            }

            _In.fill_(input.data, N);

            _MatMul.forward(
                _Out.data,
                _In.data,
                _Weight.data,
                _Bias != null ? _Bias.data : null,
                B,
                I,
                O);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            if (_Weight == null) throw new ObjectDisposedException(GetType().FullName);

            uint N = output.numel();

            uint B = (N + O - 1) / O;

            if (_Out.numel() != B * O || _In.numel() != B * I)
                throw new ArgumentOutOfRangeException(nameof(output));

            _In.zero_grad();

            _MatMul.backward(
                output.data,
                output.grad,
                _In.data,
                _In.grad,
                _Weight.data,
                _Weight.grad,
                _Bias != null ? _Bias.data : null,
                _Bias != null ? _Bias.grad : null,
                B,
                I,
                O);

            return _In;
        }
    }
}