cls
del MatMul.obj
cl.exe /c /Gz /O2 /Ot /fp:fast /Qvec-report:2 /arch:AVX2 MatMul.c
dumpbin /DISASM:BYTES MatMul.obj > MatMul.AVX2.asm
..\..\tools\coff\bin\x64\Debug\coff.exe MatMul.obj  > MatMul.AVX2.txt
del MatMul.obj
