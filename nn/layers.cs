namespace nn {
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;

    public interface ICompute {
        Tensor forward(Tensor input);
        Tensor backward(Tensor output);
    }

    public interface IModel : ICompute, IDisposable {
        IEnumerable<Tensor> parameters();
        void eval();
        void train();
    }

    public sealed class Identity : IModel {
        public void Dispose() { }
        public IEnumerable<Tensor> parameters() { yield break; }
        public void eval() { }
        public void train() { }
        public Tensor forward(Tensor input) { return input; }
        public Tensor backward(Tensor output) { return output; }
    }

    public unsafe sealed class Sequential : IModel, IEnumerable<IModel> {
        IModel[] _Mx = Array.Empty<IModel>();

        public IEnumerator<IModel> GetEnumerator() {
            return ((IEnumerable<IModel>)_Mx).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator() {
            return _Mx.GetEnumerator();
        }

        public IModel this[int index] {
            get {
                return _Mx[index];
            }
        }

        public Sequential(params IModel[] modules) {
            foreach (var m in modules) {
                add(m);
            }
        }

        public void add(IModel m) {
            var Mx = new IModel[_Mx.Length + 1];
            Array.Copy(_Mx, Mx, _Mx.Length);
            Mx[_Mx.Length] = m;
            _Mx = Mx;
        }

        public void Dispose() {
            var M = _Mx;
            _Mx = null;
            for (int m = M.Length - 1; m >= 0; m--) {
                M[m].Dispose();
            }
        }

        public IEnumerable<Tensor> parameters() {
            foreach (IModel m in _Mx) {
                foreach (var p in m.parameters()) {
                    yield return p;
                }
            }
        }

        public void eval() {
            foreach (IModel m in _Mx) {
                m.eval();
            }
        }

        public void train() {
            foreach (IModel m in _Mx) {
                m.train();
            }
        }

        public Tensor forward(Tensor input) {
            for (int m = 0; m < _Mx.Length; m++) {
                input = _Mx[m].forward(input);
            }
            return input;
        }

        public Tensor backward(Tensor output) {
            for (int m = _Mx.Length - 1; m >= 0; m--) {
                output = _Mx[m].backward(output);
            }
            return output;
        }
    }

    public unsafe class Tanh : IModel {
        Tensor _In, _Out;

        public void Dispose() {
            _In = null;
            if (_Out != null) _Out.Dispose();
            _Out = null;
        }

        public IEnumerable<Tensor> parameters() { yield break; }
        public void eval() { }
        public void train() { }

        public Tensor forward(Tensor input) {
            if (input is null) {
                throw new ArgumentNullException(nameof(input));
            }

            uint N = input.numel();

            _In = input;

            if (_Out is null) {
                _Out = new Tensor(N, requires_grad: true);
            } else {
                if (_Out.numel() != N) {
                    _Out.resize(N);
                }
            }

            F.tanh_forward_cpu(
                _Out.data,
                _In.data,
                N);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            if (output is null) {
                throw new ArgumentNullException(nameof(output));
            }

            uint N = output.numel();

            if (_In.numel() != N) throw new InvalidOperationException();

            _In.zero_grad();

            F.tanh_backward_cpu(
                output.data,
                output.grad,
                _In.data,
                _In.grad,
                N);

            return _In;
        }
    }

    public unsafe class ReLU : IModel {
        Tensor _In, _Out;

        public void Dispose() {
            _In = null;
            if (_Out != null) _Out.Dispose();
            _Out = null;
        }

        public IEnumerable<Tensor> parameters() { yield break; }
        public void eval() { }
        public void train() { }

        public Tensor forward(Tensor input) {
            if (input is null) {
                throw new ArgumentNullException(nameof(input));
            }

            uint N = input.numel();

            _In = input;

            if (_Out is null) {
                _Out = new Tensor(N, requires_grad: true);
            } else {
                if (_Out.numel() != N) {
                    _Out.resize((N));
                }
            }

            F.relu_forward_cpu(
                _Out.data,
                _In.data,
                N);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            if (output is null) {
                throw new ArgumentNullException(nameof(output));
            }

            uint N = output.numel();

            if (_In.numel() != N) throw new InvalidOperationException();

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

    public unsafe class LeakyReLU : IModel {
        public readonly double negval;

        Tensor _In, _Out;

        public LeakyReLU(double negval = 1e-2) {
            this.negval = negval;
        }

        public void Dispose() {
            _In = null;
            if (_Out != null) _Out.Dispose();
            _Out = null;
        }

        public IEnumerable<Tensor> parameters() { yield break; }
        public void eval() { }
        public void train() { }

        public Tensor forward(Tensor input) {
            if (input is null) {
                throw new ArgumentNullException(nameof(input));
            }

            uint N = input.numel();

            _In = input;

            if (_Out is null) {
                _Out = new Tensor(N, requires_grad: true);
            } else {
                if (_Out.numel() != N) {
                    _Out.resize((N));
                }
            }

            F.leaky_relu_forward_cpu(
                _Out.data,
                _In.data,
                N,
                negval);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            if (output is null) {
                throw new ArgumentNullException(nameof(output));
            }

            uint N = output.numel();

            if (_In.numel() != N) throw new InvalidOperationException();

            _In.zero_grad();

            F.leaky_relu_backward_cpu(
                output.data,
                output.grad,
                _In.data,
                _In.grad,
                N,
                negval);

            return _In;
        }
    }

    public unsafe class Dropout : IModel {
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
            _In = null;
            if (_Out != null) _Out.Dispose();
            _Out = null;
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
            if (input is null) {
                throw new ArgumentNullException(nameof(input));
            }

            uint N = input.numel();

            if (_Out is null) {
                _Out = new Tensor(N, requires_grad: true);
            } else {
                if (_Out.numel() != N) {
                    _Out.resize(N);
                }
            }
            if (_Mask is null) {
                _Mask = new Tensor(N, requires_grad: false);
            } else {
                if (_Mask.numel() != N) {
                    _Mask.resize(N);
                }
            }

            _In = input;

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
            if (output is null) {
                throw new ArgumentNullException(nameof(output));
            }

            uint N = output.numel();

            if (_In.numel() != N) throw new InvalidOperationException();

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

    public unsafe class Sigmoid : IModel {
        Tensor _In;
        Tensor _Out;

        public void Dispose() {
            _In = null;
            if (_Out != null) _Out.Dispose();
            _Out = null;
        }

        public IEnumerable<Tensor> parameters() { yield break; }
        public void eval() { }
        public void train() { }

        public Tensor forward(Tensor input) {
            if (input is null) {
                throw new ArgumentNullException(nameof(input));
            }

            uint N = input.numel();

            if (_Out is null) {
                _Out = new Tensor(N, requires_grad: true);
            } else {
                if (_Out.numel() != N) {
                    _Out.resize((N));
                }
            }

            _In = input;

            F.sigmoid_forward_cpu(
                _Out.data,
                _In.data,
                N);

            return _Out;
        }

        public Tensor backward(Tensor output) {
            if (output is null) {
                throw new ArgumentNullException(nameof(output));
            }

            uint N = output.numel();

            if (_In.numel() != N) throw new InvalidOperationException();

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
    public unsafe class Linear : IModel {
        F.MatMul _MatMul;

        public readonly uint I;
        public readonly uint O;

        Tensor _In;     /* [B, I] */
        Tensor _Out;    /* [B, O] */

        public readonly Tensor _Weight; /* [I, O] */
        public readonly Tensor _Bias;   /* [O] */

        public Linear(
            int I,
            int O,
            bool bias = true,
            int maxDegreeOfParallelism = -1,
            bool naive = false) {

            if (I <= 0 || I >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(I));
            }
            if (O <= 0 || O >= short.MaxValue / 2) {
                throw new ArgumentOutOfRangeException(nameof(O));
            }

            this.I = (uint)I;
            this.O = (uint)O;

            _Weight = new Tensor(this.O * this.I, requires_grad: true);

            _Bias = bias
                ? new Tensor(this.O, requires_grad: true)
                : null;

            if (!naive && System.Runtime.Intrinsics.X86.Avx2.IsSupported) {
                _MatMul = new nn.CPU.MatMulAVX2(maxDegreeOfParallelism);
            } else {
                _MatMul = new nn.F.MatMul(maxDegreeOfParallelism);
            }
        }

        public override string ToString() {
            return $"{GetType().FullName}<{(_MatMul != null ? _MatMul.GetType().Name : "")}>";
        }

        public void Dispose() {
            _In = null;
            if (_Out != null)
                _Out.Dispose();
            _Out = null;
            if (_Bias != null)
                _Bias.Dispose();
            if (_Weight != null)
                _Weight.Dispose();
            if (_MatMul != null)
                _MatMul.Dispose();
            _MatMul = null;
        }

        public IEnumerable<Tensor> parameters() {
            if (_Weight != null) yield return _Weight;
            if (_Bias != null) yield return _Bias;
        }

        public void eval() { }
        public void train() { }

        public Tensor forward(Tensor input) {
            if (_MatMul == null) throw new ObjectDisposedException(GetType().FullName);

            if (input is null) {
                throw new ArgumentNullException(nameof(input));
            }

            uint N = input.numel();

            uint B = (N + I - 1) / I;

            if (B * I != N) throw new ArgumentOutOfRangeException(nameof(input));

            if (_Out is null) {
                _Out = new Tensor(B * O, requires_grad: true);
            } else {
                if (_Out.numel() != B * O) {
                    _Out.resize(B * O);
                }
            }

            _In = input;

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
            if (_MatMul == null) throw new ObjectDisposedException(GetType().FullName);

            if (output is null) {
                throw new ArgumentNullException(nameof(output));
            }

            uint N = output.numel();

            uint B = (N + O - 1) / O;

            if (_Out.numel() != B * O || _In.numel() != B * I) throw new ArgumentOutOfRangeException(nameof(output));

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