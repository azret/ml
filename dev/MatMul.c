/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /Gz /Ot /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev MatMul.c
//            cl.exe /Gz /Ot /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev MatMul.c
//            cl.exe /Gz /Ot /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev MatMul.c
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include <corecrt_math_defines.h>
#include "rand.h"

#define CLOCK_MONOTONIC 0

inline int clock_gettime(int ignore_variable, struct timespec* tv) { return timespec_get(tv, TIME_UTC); }
void validate_results_cpu(const float* device_result, const float* cpu_reference, const char* name, int num_elements, float tolerance);
float* make_random_float(size_t N);

// Naive reference implementation
void matmul_forward_cpu_c_0(
    float* _Out,       /* [B, O] */
    float* _In,        /* [B, I] */
    float* _Weight,    /* [I, O] */
    float* _Bias,      /* [O] */
    unsigned int B,
    unsigned int I,
    unsigned int O) {

    for (unsigned int b = 0; b < B; b++) {
        for (unsigned int o = 0; o < O; o++) {
            _Out[b * O + o] = _Bias ? _Bias[o] : 0;
            for (unsigned int i = 0; i < I; i++) {
                _Out[b * O + o] += _Weight[o * I + i] * _In[b * I + i];
            }
        }
    }
}

void matmul_forward_cpu_c_1(
    float* _Out,       /* [B, O] */
    float* _In,        /* [B, I] */
    float* _Weight,    /* [I, O] */
    float* _Bias,      /* [O] */
    unsigned int B,
    unsigned int I,
    unsigned int O) {

    for (unsigned int b = 0; b < B; b++) {
        float* x = _In + b * I;
        float* y = _Out + b * O;
        for (unsigned int o = 0; o < O; o++) {
            float acc = _Bias ? _Bias[o] : 0;
            float* w = _Weight + o * I;
            for (unsigned int i = 0; i < I; i++) {
                acc += w[i] * x[i];
            }
            y[o] = acc;
        }
    }
}


#define MATMUL_FORWARD_KERNELS 2

void matmul_forward_cpu(
    int kernel_num,
    float* _Out,       /* [B, O] */
    float* _In,        /* [B, I] */
    float* _Weight,    /* [I, O] */
    float* _Bias,      /* [O] */
    unsigned int B,
    unsigned int I,
    unsigned int O) {

    switch (kernel_num) {
    case 0:
        matmul_forward_cpu_c_0(_Out, _In, _Weight, _Bias, B, I, O);
        break;
    case 1:
        matmul_forward_cpu_c_1(_Out, _In, _Weight, _Bias, B, I, O);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}


// Naive reference implementation
void matmul_backward_cpu_0(
    float* _Out,       /* [B, O] */
    float* d_Out,       /* [B, O] */
    float* _In,        /* [B, I] */
    float* d_In,        /* [B, I] */
    float* _Weight,    /* [I, O] */
    float* d_Weight,    /* [I, O] */
    float* _Bias,      /* [O] */
    float* d_Bias,      /* [O] */
    unsigned int B,
    unsigned int I,
    unsigned int O) {

    for (unsigned int b = 0; b < B; b++) {
        for (unsigned int o = 0; o < O; o++) {
            float δf = d_Out[b * O + o];
            for (unsigned int i = 0; i < I; i++) {
                d_In[b * I + i] += _Weight[o * I + i] * δf;
            }
        }
    }

    for (unsigned int b = 0; b < B; b++) {
        for (unsigned int o = 0; o < O; o++) {
            float δf = d_Out[b * O + o];
            for (unsigned int i = 0; i < I; i++) {
                d_Weight[o * I + i] += _In[b * I + i] * δf;
            }
            if (d_Bias) {
                d_Bias[o] += δf;
            }
        }
    }
}

// Single pass
void matmul_backward_cpu_1(
    float* _Out,       /* [B, O] */
    float* d_Out,       /* [B, O] */
    float* _In,        /* [B, I] */
    float* d_In,        /* [B, I] */
    float* _Weight,    /* [I, O] */
    float* d_Weight,    /* [I, O] */
    float* _Bias,      /* [O] */
    float* d_Bias,      /* [O] */
    unsigned int B,
    unsigned int I,
    unsigned int O) {

    for (unsigned int b = 0; b < B; b++) {
        float* p_In = _In + b * I;
        float* p_d_In = d_In + b * I;
        for (unsigned int o = 0; o < O; o++) {
            float δf = d_Out[b * O + o];
            float* p_Weight = _Weight + o * I;
            float* p_d_Weight = d_Weight + o * I;
            for (unsigned int i = 0; i < I; i++) {
                p_d_In[i] += p_Weight[i] * δf;
                p_d_Weight[i] += p_In[i] * δf;
            }
            if (d_Bias) {
                d_Bias[o] += δf;
            }
        }
    }
}

#define MATMUL_BACKWARD_KERNELS 2

void matmul_backward_cpu(
    int kernel_num,
    float* _Out,       /* [B, O] */
    float* d_Out,       /* [B, O] */
    float* _In,        /* [B, I] */
    float* d_In,        /* [B, I] */
    float* _Weight,    /* [I, O] */
    float* d_Weight,    /* [I, O] */
    float* _Bias,      /* [O] */
    float* d_Bias,      /* [O] */
    unsigned int B,
    unsigned int I,
    unsigned int O) {

    switch (kernel_num) {
        case 0:
            matmul_backward_cpu_0(_Out, d_Out, _In, d_In, _Weight, d_Weight, _Bias, d_Bias, B, I, O);
            break;
        case 1:
            matmul_backward_cpu_1(_Out, d_Out, _In, d_In, _Weight, d_Weight, _Bias, d_Bias, B, I, O);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


#include "intrin.h"
#include "assert.h"

#define MEMCPY_KERNELS 2

void memcopy_1(float* _Dst, float* _Src, size_t Size) {
    assert(Size % 32 == 0);
    while (Size) {
        _mm256_store_si256((__m256i*)_Dst, _mm256_load_si256((__m256i const*)_Src));
        _Src += 32;
        _Dst += 32;
        Size -= 32;
    }
}

void mem_copy(
    int kernel_num,
    float* _Dst,
    float* _Src,
    size_t Len) {

    switch (kernel_num) {
    case 0:
        memcpy(_Dst, _Src, Len * sizeof(float));
        break;
    case 1:
        memcopy_1(_Dst, _Src, Len * sizeof(float));
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

int main_MatMul(int argc, char **argv) {

    int B = 64;
    int C = 1024;
    int OC = 1024 * 4;

    int RUNS = 32; // number of times to run a kernel for benchmarks

    srand(137);

    float* out = make_random_float(B * OC);
    float* inp = make_random_float(B * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    float* d_out = make_random_float(B * OC);
    float* d_inp = make_random_float(B * C);
    float* d_weight = make_random_float(OC * C);
    float* d_bias = make_random_float(OC);

    printf("> Calculating reference\n");

    matmul_forward_cpu_c_0(
        out,
        inp,
        weight,
        bias,
        B,
        C,
        OC);

    matmul_backward_cpu_0(
        out,
        d_out,
        inp,
        d_inp,
        weight,
        d_weight,
        bias,
        d_bias,
        B,
        C,
        OC);

    for (int kernel_num = 0; kernel_num < MATMUL_FORWARD_KERNELS; kernel_num++) {
        printf("> Verifying \033[0;36mmatmul_forward kernel #%d\033[0m\n", kernel_num);

        srand(137);

        float* kernel_out = make_random_float(B * OC);
        float* kernel_inp = make_random_float(B * C);
        float* kernel_weight = make_random_float(OC * C);
        float* kernel_bias = make_random_float(OC);

        float* kernel_d_out = make_random_float(B * OC);
        float* kernel_d_inp = make_random_float(B * C);
        float* kernel_d_weight = make_random_float(OC * C);
        float* kernel_d_bias = make_random_float(OC);

        matmul_forward_cpu(kernel_num, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * OC, 1e-5f);

        free(kernel_d_out);
        free(kernel_d_inp);
        free(kernel_d_weight);
        free(kernel_d_bias);

        free(kernel_out);
        free(kernel_inp);
        free(kernel_weight);
        free(kernel_bias);
    }

    for (int kernel_num = 0; kernel_num < MATMUL_BACKWARD_KERNELS; kernel_num++) {
        printf("> Verifying \033[0;36mmatmul_backward kernel #%d\033[0m\n", kernel_num);

        srand(137);

        float* kernel_out = make_random_float(B * OC);
        float* kernel_inp = make_random_float(B * C);
        float* kernel_weight = make_random_float(OC * C);
        float* kernel_bias = make_random_float(OC);

        float* kernel_d_out = make_random_float(B * OC);
        float* kernel_d_inp = make_random_float(B * C);
        float* kernel_d_weight = make_random_float(OC * C);
        float* kernel_d_bias = make_random_float(OC);

        matmul_backward_cpu(kernel_num,
            kernel_out,
            kernel_d_out,
            kernel_inp,
            kernel_d_inp,
            kernel_weight,
            kernel_d_weight,
            kernel_bias,
            kernel_d_bias,
            B,
            C,
            OC);

        validate_results_cpu(kernel_d_inp, d_inp, "d_inp", B * C, 1e-5f);
        validate_results_cpu(kernel_d_weight, d_weight, "d_weight", OC * C, 1e-5f);
        validate_results_cpu(kernel_d_bias, d_bias, "d_bias", OC, 1e-5f);

        free(kernel_d_out);
        free(kernel_d_inp);
        free(kernel_d_weight);
        free(kernel_d_bias);

        free(kernel_out);
        free(kernel_inp);
        free(kernel_weight);
        free(kernel_bias);
    }

    printf("\nAll kernels passed! Starting benchmarks.\n\n");


    //---------------------------------------

    for (int kernel_num = 0; kernel_num < MATMUL_FORWARD_KERNELS; kernel_num++) {
        printf("> Running matmul_forward kernel #%d\n", kernel_num);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int i = 0; i < RUNS; i++) {
            matmul_forward_cpu(kernel_num, out, inp, weight, bias, B, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf(">  \033[0;36mmatmul_forward #%d \033[0m, (took \033[0;33m%f\033[0m ms)\n", kernel_num, time_elapsed_s * 1000);
    }

    //---------------------------------------

    for (int kernel_num = 0; kernel_num < MATMUL_BACKWARD_KERNELS; kernel_num++) {
        printf("> Running matmul_backward kernel #%d\n", kernel_num);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int i = 0; i < RUNS; i++) {
            matmul_backward_cpu(kernel_num,
                out,
                d_out,
                inp,
                d_inp,
                weight,
                d_weight,
                bias,
                d_bias,
                B,
                C,
                OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf(">  \033[0;36mmatmul_backward #%d \033[0m, (took \033[0;33m%f\033[0m ms)\n", kernel_num, time_elapsed_s * 1000);
    }

    //---------------------------------------

    float* dst = (float*)malloc(1024 * 1024 * 1024 * sizeof(float));
    float* src = (float*)malloc(1024 * 1024 * 1024 * sizeof(float));

    for (int kernel_num = 0; kernel_num < MEMCPY_KERNELS; kernel_num++) {
        printf("> Running memcpy kernel #%d\n", kernel_num);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int i = 0; i < RUNS; i++) {
            mem_copy(
                kernel_num,
                dst,
                src,
                1024 * 1024 * 1024);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf(">  \033[0;36mmemcpy #%d \033[0m, (took \033[0;33m%f\033[0m ms)\n", kernel_num, time_elapsed_s * 1000);
    }

    free(dst);
    free(src);

    //---------------------------------------

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);

    free(d_out);
    free(d_inp);
    free(d_weight);
    free(d_bias);

    printf("\nDone");
    getchar();

    return 0;
}


int main(int argc, char** argv) {

    mt19937_state state;
    manual_seed(&state, 137);
    printf("%u\n", randint32(&state));
    printf("%u\n", randint32(&state));
    printf("%u\n", randint32(&state));
    printf("%u\n", randint32(&state));
    printf("%u\n", randint32(&state));

    float t8[8];
    normal_(t8, 8, 0, 1, &state);
    for (int i = 0; i < 8; i++) {
        printf("%f\n", t8[i]);
    }
    printf("%u\n", randint32(&state));

    float t16[16];
    normal_(t16, 16, 0, 1, &state);
    for (int i = 0; i < 16; i++) {
        printf("%f\n", t16[i]);
    }
    printf("%u\n", randint32(&state));

    getchar();
    exit(0);


    //---------------------------------------

    float* dst = (float*)malloc(1024 * 1024 * 1024 * sizeof(float));
    float* src = (float*)malloc(1024 * 1024 * 1024 * sizeof(float));

    for (int kernel_num = 0; kernel_num < MEMCPY_KERNELS; kernel_num++) {
        printf("> Running memcpy kernel #%d\n", kernel_num);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int i = 0; i < 64; i++) {
            mem_copy(
                kernel_num,
                dst,
                src,
                1024 * 1024 * 1024);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf(">  \033[0;36mmemcpy #%d \033[0m, (took \033[0;33m%f\033[0m ms)\n", kernel_num, time_elapsed_s * 1000);
    }

    free(dst);
    free(src);

    printf("\nDone");
    getchar();

    return 0;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // range -1..1
    }
    return arr;
}

void validate_results_cpu(const float* kernel_result, const float* cpu_reference, const char* name, int num_elements, float tolerance) {
    int nfaults = 0;
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        float t_eff = tolerance + fabsf(cpu_reference[i]);
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("\033[1;33mMismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\033[0m\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            if (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    if (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("\033[1;32mOK\033[0m\n");
}