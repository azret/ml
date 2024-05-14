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

        public IEnumerable<Tensor> parameters() {
            yield break;
        }

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

    public unsafe sealed class Sigmoid : ILayer {
        Tensor _In, _Out;

        public void Dispose() {
            if (_Out != null)
                _Out.Dispose();
            _Out = null;
            if (_In != null)
                _In.Dispose();
            _In = null;
        }

        public IEnumerable<Tensor> parameters() {
           yield break;
        }

        public Tensor forward(Tensor input) {
            uint N = input.numel();

            // Dynamically create space for input and output to accommodate the batch size

            if (_In is null) {
                _In = new Tensor(N, requires_grad: true);
            } else {
                if (_In.numel() != N) {
                    _In.numel((N));
                }
            }
            if (_Out is null) {
                _Out = new Tensor(N, requires_grad: true);
            } else {
                if (_Out.numel() != N) {
                    _Out.numel((N));
                }
            }

            _In.memcpy(input.data, N);

            F.sigmoid_forward_cpu(
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
                output,
                _In,
                N);

            return _In;
        }
    }


    [DebuggerDisplay("({_in_features}, {_out_features})")]
    public unsafe class Linear : ILayer {
        Tensor _In, _Out;

        public readonly uint _in_features;
        public readonly uint _out_features;

        public readonly Tensor _Weight;
        public readonly Tensor _Bias;

        public Linear(uint in_features, uint out_features, bool bias = true) {
            if (in_features <= 0 || in_features >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(in_features));
            }

            if (out_features <= 0 || out_features >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(out_features));
            }

            _in_features = in_features;
            _out_features = out_features;

            _Weight = new Tensor(checked(out_features * in_features), requires_grad: true);

            _Bias = bias
                ? new Tensor(checked(out_features), requires_grad: true)
                : null;
        }

        public void Dispose() {
            if (_Out != null)
                _Out.Dispose();
            _Out = null;

            if (_In != null)
                _In.Dispose();
            _In = null;

            if (_Bias != null)
                _Bias.Dispose();

            if (_Weight != null) 
                _Weight.Dispose();
        }

        public IEnumerable<Tensor> parameters() {

            // Optimized parameters

            if (_Weight != null) yield return _Weight;
            if (_Bias != null) yield return _Bias;
        }

        public Tensor forward(Tensor input) {
            uint N = input.numel();

            uint B = checked(N + _in_features - 1) / _in_features;

            if (checked(B * _in_features) != N) throw new ArgumentOutOfRangeException(nameof(input));

            // Dynamically create space for input and output to accommodate the batch size

            if (_In is null) {
                _In = new Tensor(checked(B * _in_features), requires_grad: true);
            } else {
                if (_In.numel() != checked(B * _in_features)) {
                    _In.numel(checked(B * _in_features));
                }
            }

            if (_Out is null) {
                _Out = new Tensor(checked(B * _out_features), requires_grad: true);
            } else {
                if (_Out.numel() != checked(B * _out_features)) {
                    _Out.numel(checked(B * _out_features));
                }
            }

            _In.memcpy(input.data, N);

            F.matmul_forward_cpu(
                _Out.data,
                _In.data,
                _Weight.data,
                _Bias.data,
                B,
                _in_features,
                _out_features);

            return _Out;
        }

        public Tensor backward(Tensor output) {

            uint B = checked(output.numel() + _out_features - 1) / _out_features;

            if (_Out.numel() != checked(B * _out_features) || _In.numel() != checked(B * _in_features))
                throw new ArgumentOutOfRangeException(nameof(output));

            _In.zero_grad();

            F.matmul_backward_cpu(
                output,
                _In,
                _Weight,
                _Bias,
                B,
                _in_features,
                _out_features);

            return _In;
        }
    }

}