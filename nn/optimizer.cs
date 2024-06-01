namespace nn {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using static std;

    public interface IOptimizer {
        uint get_num_params();
        void reset();
        void step();
        void update();
        void zero_grad();
    }

    public abstract class Optimizer : IOptimizer {
        protected ulong _step = 0;

        protected float _lr;

        protected Tensor[] _params;

        public Optimizer(IEnumerable<Tensor> all_params, float lr) {
            _lr = lr;
            _params = all_params.ToArray();
        }

        public void zero_grad() {
            for (int p = 0; p < _params.Length; p++) {
                _params[p].zero_grad();
            }
        }

        public virtual void reset() {
            _step = 0;
        }

        public void step() {
            update();
            _step++;
        }

        public abstract void update();

        public uint get_num_params() {
            uint num = 0;

            for (int p = 0; p < _params.Length; p++) {
                Tensor tensor = _params[p];
                num += tensor.numel();
            }

            return num;
        }
    }

    public unsafe class SGD : Optimizer {

        // reference: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html

        public SGD(IEnumerable<Tensor> parameters, float lr = 1e-3f)
            : base(parameters, lr) {
        }

        public override void update() {
            for (int p = 0; p < _params.Length; p++) {
                Tensor T = _params[p];

                for (int n = 0; n < T.numel(); n++) {
                    T.data[n] -= T.grad[n] * _lr;
                }
            }
        }
    }

    public unsafe class AdamW : Optimizer, IDisposable {

        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        float _beta1 = 0.9f;
        float _beta2 = 0.999f;
        float _eps = 1e-8f;
        float _weight_decay = 0.01f;

        float* m_memory;
        float* v_memory;

        public AdamW(
            IEnumerable<Tensor> parameters,
            float lr = 1e-3f,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float eps = 1e-8f,
            float weight_decay = 0.01f) : base(parameters, lr) {

            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _weight_decay = weight_decay;
            _params = parameters.ToArray();
        }

        public void Dispose() {
            free(m_memory);
            m_memory = null;
            free(v_memory);
            v_memory = null;
        }

        public override void reset() {
            base.reset();
            free(m_memory); m_memory = null;
            free(v_memory); v_memory = null;
        }

        public override void update() {
            uint i = 0, p;

            if (m_memory == null) {
                for (p = 0; p < _params.Length; p++) {
                    Tensor tensor = _params[p];
                    i += (uint)tensor.numel();
                }

                m_memory = (float*)malloc(i * (uint)sizeof(float));
                v_memory = (float*)malloc(i * (uint)sizeof(float));

                memset(m_memory, 0, i * sizeof(float));
                memset(v_memory, 0, i * sizeof(float));
            }

            for (p = 0, i = 0; p < _params.Length; p++) {
                Tensor T = _params[p];

                for (int n = 0; n < T.numel(); n++, i++) {
                    double δf  = T.grad[n];

                    double m = _beta1 * m_memory[i] + (1.0f - _beta1) * δf;
                    double v = _beta2 * v_memory[i] + (1.0f - _beta2) * δf * δf;

                    double m_hat = m / (1.0 - Math.Pow(_beta1, _step + 1));
                    double v_hat = v / (1.0 - Math.Pow(_beta2, _step + 1));

                    m_memory[i] = (float)m;
                    v_memory[i] = (float)v;

                    T.data[n] -= (float)(_lr * (m_hat / (Math.Sqrt(v_hat) + _eps) - (double)_weight_decay * T.data[n]));
                }
            }
        }
    }
}