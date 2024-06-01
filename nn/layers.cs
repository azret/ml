namespace nn {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

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

    [DebuggerDisplay("({_in_features}, {_out_features})")]
    public unsafe class Linear : ILayer {
        Tensor _In, _Out;

        public readonly uint _in_features;
        public readonly uint _out_features;

        public readonly Tensor weight;
        public readonly Tensor bias;

        public Linear(int in_features, int out_features, bool use_bias = true) {
            if (in_features <= 0 || in_features >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(in_features));
            }
            if (out_features <= 0 || out_features >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(out_features));
            }
            _in_features = (uint)in_features;
            _out_features = (uint)out_features;
            weight = new Tensor(checked(_out_features * _in_features), requires_grad: true);
            bias = use_bias
                ? new Tensor(_out_features, requires_grad: true)
                : null;
        }

        public void Dispose() {
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

        protected virtual void forward(
            float* _Out,       /* [B, O] */
            float* _In,        /* [B, I] */
            float* _Weight,    /* [I, O] */
            float* _Bias,      /* [O] */
            uint B,
            uint I,
            uint O) {

            F.matmul_forward_cpu(
                _Out,
                _In,
                _Weight,
                _Bias,
                B,
                I,
                O);
        }

        protected virtual void backward(
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

            F.matmul_backward_cpu(
                _Out,
                d_Out,
                _In,
                d_In,
                _Weight,
                d_Weight,
                _Bias,
                d_Bias,
                B,
                I,
                O);
        }

        public Tensor forward(Tensor input) {
            uint N = input.numel();

            // Dynamically create space for input and output to accommodate the batch size

            uint B = (N + _in_features - 1) / _in_features;

            if (B * _in_features != N) throw new ArgumentOutOfRangeException(nameof(input));

            if (_In is null) {
                _In = new Tensor(N, requires_grad: true);
            } else {
                if (_In.numel() != B * _in_features) {
                    _In.resize(B * _in_features);
                }
            }

            if (_Out is null) {
                _Out = new Tensor(B * _out_features, requires_grad: true);
            } else {
                if (_Out.numel() != B * _out_features) {
                    _Out.resize(B * _out_features);
                }
            }

            _In.memcpy(input.data, N);

            forward(
                _Out.data,
                _In.data,
                weight.data,
                bias != null ? bias.data : null,
                B,
                _in_features,
                _out_features);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            uint N = output.numel();

            uint B = (N + _out_features - 1) / _out_features;

            if (_Out.numel() != B * _out_features || _In.numel() != B * _in_features)
                throw new ArgumentOutOfRangeException(nameof(output));

            _In.zero_grad();

            backward(
                output.data,
                output.grad,
                _In.data,
                _In.grad,
                weight.data,
                weight.grad,
                bias != null ? bias.data : null,
                bias != null ? bias.grad : null,
                B,
                _in_features,
                _out_features);

            return _In;
        }
    }

}