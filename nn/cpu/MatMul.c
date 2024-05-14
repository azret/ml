// cl /c /Gz /O2 /fp:fast /Qvec-report:2 MatMul.c

// cl /c /Gz /O2 /fp:fast /arch:AVX /Qvec-report:2 MatMul.c

// cl /c /Gz /O2 /fp:fast /arch:AVX2 /Qvec-report:2 MatMul.c

// dumpbin /DISASM:BYTES MatMul.obj > MatMul.asm

void matmul_forward_cpu_c(
    float* _Out,       /* [B, O] */
    float* _In,        /* [B, I] */
    float* _Weight,    /* [I, O] */
    float* _Bias,      /* [O] */
    int B,
    int I,
    int O) {

    for (int b = 0; b < B; b++) {
        for (int o = 0; o < O; o++) {
            _Out[b * O + o] = _Bias ? _Bias[o] : 0;
            for (int i = 0; i < I; i++) {
                _Out[b * O + o] += _Weight[o * I + i] * _In[b * I + i];
            }
        }
    }
}