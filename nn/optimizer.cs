namespace nn {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using static std;

    public interface IOptimizer : IDisposable {
        float lr { get; set; }
        float momentum { get; set; }
        float weight_decay { get; set; }
        uint get_num_params();
        void reset();
        void step();
        void update();
        void zero_grad();
    }

    public abstract class Optimizer : IOptimizer {
        protected ulong _step = 0;

        float _lr;
        float _momentum;
        float _weight_decay;

        protected Tensor[] _params;

        public Optimizer(IEnumerable<Tensor> all_params, float lr, float momentum, float weight_decay) {
            _lr = lr;
            _momentum = momentum;
            _weight_decay = weight_decay;
            _params = all_params.ToArray();
        }

        public float lr { get => _lr; set => _lr = value; }
        public float momentum { get => _momentum; set => _momentum = value; }
        public float weight_decay { get => _weight_decay; set => _weight_decay = value; }

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

        public abstract void Dispose();

        public uint get_num_params() {
            uint num_params = 0;
            for (int p = 0; p < _params.Length; p++) {
                Tensor param = _params[p];
                num_params += param.numel();
            }
            return num_params;
        }
    }

    public unsafe class SGD : Optimizer {

        // reference: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html

        float _dampening;

        float* m_buffer;

        public SGD(IEnumerable<Tensor> parameters, float lr = 1e-3f, float momentum = 0.0f, float weight_decay = 0.0f, float dampening = 0.0f)
            : base(parameters,
                  lr: lr,
                  momentum: momentum,
                  weight_decay: weight_decay) {
            _dampening = dampening;
        }

        public override void Dispose() {
            free(m_buffer); m_buffer = null;
        }

        public override void reset() {
            base.reset();
            free(m_buffer); m_buffer = null;
        }

        public override void update() {
            var val_momentum = momentum;
            var val_weight_decay = weight_decay;
            if (val_momentum != 0 && m_buffer == null) {
                uint num_params = 0;
                for (uint i = 0; i < _params.Length; i++) {
                    Tensor param = _params[i];
                    num_params += param.numel();
                }
                m_buffer = (float*)malloc(num_params * sizeof(float));
                memset(m_buffer, 0, num_params * sizeof(float));
            }
            int m = -1;
            for (int i = 0; i < _params.Length; i++) {
                Tensor param = _params[i];
                for (int n = 0; n < param.numel(); n++) {
                    var d_p = param.grad[n];
                    if (val_weight_decay != 0.0f) {
                        d_p += param.data[n] * val_weight_decay;
                    }
                    if (m_buffer != null) {
                        int k = m++;
                        m_buffer[k] *= val_momentum;
                        m_buffer[k] += d_p * (1 - _dampening);
                        param.data[n] += -lr * m_buffer[k];
                    } else {
                        param.data[n] += -lr * d_p;
                    }
                }
            }
        }
    }

    public unsafe class AdamW : Optimizer, IDisposable {

        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        float _beta1;
        float _beta2;
        float _eps;

        float* m_buffer;
        float* v_buffer;

        public AdamW(IEnumerable<Tensor> parameters,
            float lr = 1e-3f,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float eps = 1e-8f,
            float weight_decay = 1e-2f)

            : base(parameters, lr: lr, momentum: 0, weight_decay: weight_decay) {

            _beta1 = beta1;
            _beta2 = beta2;
            _eps = eps;
            _params = parameters.ToArray();
        }

        public override void Dispose() {
            free(m_buffer); m_buffer = null;
            free(v_buffer); v_buffer = null;
        }

        public override void reset() {
            base.reset();
            free(m_buffer); m_buffer = null;
            free(v_buffer); v_buffer = null;
        }

        public override void update() {
            if (m_buffer == null) {
                uint num_params = 0;
                for (uint p = 0; p < _params.Length; p++) {
                    Tensor param = _params[p];
                    num_params += param.numel();
                }
                m_buffer = (float*)malloc(num_params * sizeof(float));
                v_buffer = (float*)malloc(num_params * sizeof(float));
                memset(m_buffer, 0, num_params * sizeof(float));
                memset(v_buffer, 0, num_params * sizeof(float));
            }

            var val_weight_decay = weight_decay;
            int z = -1;

            for (uint p = 0; p < _params.Length; p++) {
                Tensor T = _params[p];

                //Parallel.For(0, T.numel(), (n) => {
                    for (int n = 0; n < T.numel(); n++) 
                    {
                        var i = z++;

                        float δf = T.grad[n];

                    float m = (float)_beta1 * (float)m_buffer[i] + (1.0f - (float)_beta1) * δf;
                    float v = (float)_beta2 * (float)v_buffer[i] + (1.0f - (float)_beta2) * δf * δf;

                    float m_hat = m / (1.0f - (float)Math.Pow(_beta1, _step + 1));
                    float v_hat = v / (1.0f - (float)Math.Pow(_beta2, _step + 1));

                        m_buffer[i] = (float)m;
                        v_buffer[i] = (float)v;

                        T.data[n] -= (float)(lr * (m_hat / ((float)Math.Sqrt(v_hat) + _eps) - (float)val_weight_decay * T.data[n]));

                    }
                // });
            }
        }
    }
}