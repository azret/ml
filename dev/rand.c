#include <stdio.h>
#include <math.h>
#include <corecrt_math_defines.h>
#include "rand.h"
#include <float.h>

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
        printf("%.16f\n", t8[i]);
    }
    printf("%u\n", randint32(&state));
    
    float t16[16];
    normal_(t16, 16, 0, 1, &state);
    for (int i = 0; i < 16; i++) {
        printf("%.16f\n", t16[i]);
    }
    printf("%u\n", randint32(&state));

    getchar();

    return 0;
}
