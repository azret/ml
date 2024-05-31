cls
del MatMul.exe
cl.exe /Gz /O2 /Ot /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev MatMul.c
MatMul.exe