namespace nn {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using static nn.F;

    public interface ICompute {
        Tensor forward(Tensor input);
        Tensor backward(Tensor output);
    }

    public interface ILayer : ICompute, IDisposable {
        IEnumerable<Tensor> parameters();
    }

    public sealed class Identity : ILayer {
        public void Dispose() {
        }

        public IEnumerable<Tensor> parameters() { yield break; }

        public Tensor forward(Tensor input) {
            return input;
        }

        public Tensor backward(Tensor output) {
            return output;
        }
    }

    public unsafe sealed class Sequential : ILayer {
        ILayer[] _M = new ILayer[0];

        public Sequential(params ILayer[] modules) {
            foreach (var m in modules) {
                add(m);
            }
        }

        public void add(ILayer m) {
            var M = new ILayer[_M.Length + 1];
            Array.Copy(_M, M, _M.Length);
            M[_M.Length] = m;
            _M = M;
        }

        public void Dispose() {
            var M = _M;
            _M = null;
            for (int m = M.Length - 1; m >= 0; m--) {
                M[m].Dispose();
            }
        }

        public IEnumerable<Tensor> parameters() {
            foreach (ILayer m in _M) {
                foreach (var p in m.parameters()) {
                    yield return p;
                }
            }
        }

        public Tensor forward(Tensor input) {
            for (int m = 0; m < _M.Length; m++) {
                input = _M[m].forward(input);
            }
            return input;
        }

        public Tensor backward(Tensor output) {
            for (int m = _M.Length - 1; m >= 0; m--) {
                output = _M[m].backward(output);
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

            _In.memcpy(input.data, N);

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

    public unsafe class Sigmoid : ILayer {
        Tensor _In, _Out;

        public void Dispose() {
            if (_Out != null) _Out.Dispose();
            _Out = null;
            if (_In != null) _In.Dispose();
            _In = null;
        }

        public IEnumerable<Tensor> parameters() { yield break; }

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

            _In.memcpy(input.data, N);

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

    [DebuggerDisplay("({_in_features}, {_out_features})")]
    public unsafe class Linear<T_MatMul> : ILayer where T_MatMul: MatMul, new() {
        T_MatMul _MatMul_Impl;

        Tensor _In, _Out;

        public readonly uint I;
        public readonly uint O;

        public readonly Tensor weight;
        public readonly Tensor bias;

        public Linear(int in_features, int out_features, bool use_bias = true) {
            if (in_features <= 0 || in_features >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(in_features));
            }
            if (out_features <= 0 || out_features >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(out_features));
            }
            _MatMul_Impl = new T_MatMul();
            I = (uint)in_features;
            O = (uint)out_features;
            weight = new Tensor(checked(O * I), requires_grad: true);
            bias = use_bias
                ? new Tensor(O, requires_grad: true)
                : null;
        }

        public void Dispose() {
            if (_MatMul_Impl != null) _MatMul_Impl.Dispose();
            _MatMul_Impl = null;
            if (_Out != null) _Out.Dispose();
            _Out = null;
            if (_In != null) _In.Dispose();
            _In = null;
            if (bias != null) bias.Dispose();
            if (weight != null) weight.Dispose();
        }

        public IEnumerable<Tensor> parameters() {
            if (weight != null) yield return weight;
            if (bias != null) yield return bias;
        }

        public Tensor forward(Tensor input) {
            if (_MatMul_Impl == null) throw new ObjectDisposedException(GetType().FullName);

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

            _In.memcpy(input.data, N);

            _MatMul_Impl.forward(
                _Out.data,
                _In.data,
                weight.data,
                bias != null ? bias.data : null,
                B,
                I,
                O);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            if (_MatMul_Impl == null) throw new ObjectDisposedException(GetType().FullName);

            uint N = output.numel();

            uint B = (N + O - 1) / O;

            if (_Out.numel() != B * O || _In.numel() != B * I)
                throw new ArgumentOutOfRangeException(nameof(output));

            _In.zero_grad();

            _MatMul_Impl.backward(
                output.data,
                output.grad,
                _In.data,
                _In.grad,
                weight.data,
                weight.grad,
                bias != null ? bias.data : null,
                bias != null ? bias.grad : null,
                B,
                I,
                O);

            return _In;
        }
    }

}