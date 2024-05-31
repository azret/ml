Microsoft (R) COFF/PE Dumper Version 14.39.33523.0
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file MatMul.obj

File Type: COFF OBJECT

matmul_backward:
  0000000000000000: 48 8B C4           mov         rax,rsp
  0000000000000003: 4C 89 48 20        mov         qword ptr [rax+20h],r9
  0000000000000007: 4C 89 40 18        mov         qword ptr [rax+18h],r8
  000000000000000B: 48 89 50 10        mov         qword ptr [rax+10h],rdx
  000000000000000F: 53                 push        rbx
  0000000000000010: 57                 push        rdi
  0000000000000011: 48 83 EC 38        sub         rsp,38h
  0000000000000015: 49 8B F8           mov         rdi,r8
  0000000000000018: 48 8B DA           mov         rbx,rdx
  000000000000001B: 45 33 C0           xor         r8d,r8d
  000000000000001E: 44 89 04 24        mov         dword ptr [rsp],r8d
  0000000000000022: 44 39 84 24 90 00  cmp         dword ptr [rsp+0000000000000090h],r8d
                    00 00
  000000000000002A: 0F 86 F0 02 00 00  jbe         0000000000000320
  0000000000000030: 8B 94 24 A0 00 00  mov         edx,dword ptr [rsp+00000000000000A0h]
                    00
  0000000000000037: 44 8B 9C 24 98 00  mov         r11d,dword ptr [rsp+0000000000000098h]
                    00 00
  000000000000003F: 4C 8B 94 24 88 00  mov         r10,qword ptr [rsp+0000000000000088h]
                    00 00
  0000000000000047: 48 89 70 E8        mov         qword ptr [rax-18h],rsi
  000000000000004B: 4C 89 60 E0        mov         qword ptr [rax-20h],r12
  000000000000004F: 4C 89 68 D8        mov         qword ptr [rax-28h],r13
  0000000000000053: 4C 89 70 D0        mov         qword ptr [rax-30h],r14
  0000000000000057: 4C 89 78 C8        mov         qword ptr [rax-38h],r15
  000000000000005B: 0F 1F 44 00 00     nop         dword ptr [rax+rax]
  0000000000000060: 41 8B C0           mov         eax,r8d
  0000000000000063: 45 33 ED           xor         r13d,r13d
  0000000000000066: 41 0F AF C3        imul        eax,r11d
  000000000000006A: 4C 8D 24 87        lea         r12,[rdi+rax*4]
  000000000000006E: 4D 8D 0C 81        lea         r9,[r9+rax*4]
  0000000000000072: 85 D2              test        edx,edx
  0000000000000074: 0F 84 73 02 00 00  je          00000000000002ED
  000000000000007A: 41 8B C0           mov         eax,r8d
  000000000000007D: 4C 89 54 24 08     mov         qword ptr [rsp+8],r10
  0000000000000082: 0F AF C2           imul        eax,edx
  0000000000000085: 4D 8B FA           mov         r15,r10
  0000000000000088: 89 44 24 04        mov         dword ptr [rsp+4],eax
  000000000000008C: 0F 1F 40 00        nop         dword ptr [rax]
  0000000000000090: 41 03 C5           add         eax,r13d
  0000000000000093: 41 8B CD           mov         ecx,r13d
  0000000000000096: 41 0F AF CB        imul        ecx,r11d
  000000000000009A: 45 33 C0           xor         r8d,r8d
  000000000000009D: C5 FA 10 14 83     vmovss      xmm2,dword ptr [rbx+rax*4]
  00000000000000A2: 48 8B 44 24 70     mov         rax,qword ptr [rsp+70h]
  00000000000000A7: 33 DB              xor         ebx,ebx
  00000000000000A9: C5 F8 28 DA        vmovaps     xmm3,xmm2
  00000000000000AD: C4 E2 7D 18 DA     vbroadcastss ymm3,xmm2
  00000000000000B2: 4C 8D 14 88        lea         r10,[rax+rcx*4]
  00000000000000B6: 48 8B 44 24 78     mov         rax,qword ptr [rsp+78h]
  00000000000000BB: 48 8D 14 88        lea         rdx,[rax+rcx*4]
  00000000000000BF: 45 85 DB           test        r11d,r11d
  00000000000000C2: 0F 84 E0 01 00 00  je          00000000000002A8
  00000000000000C8: 41 83 FB 08        cmp         r11d,8
  00000000000000CC: 0F 82 9D 00 00 00  jb          000000000000016F
  00000000000000D2: 41 8D 43 FF        lea         eax,[r11-1]
  00000000000000D6: 48 63 C8           movsxd      rcx,eax
  00000000000000D9: 49 8D 34 8C        lea         rsi,[r12+rcx*4]
  00000000000000DD: 48 8D 04 8A        lea         rax,[rdx+rcx*4]
  00000000000000E1: 48 3B D6           cmp         rdx,rsi
  00000000000000E4: 77 09              ja          00000000000000EF
  00000000000000E6: 49 3B C4           cmp         rax,r12
  00000000000000E9: 0F 83 80 00 00 00  jae         000000000000016F
  00000000000000EF: 49 8D 3C 89        lea         rdi,[r9+rcx*4]
  00000000000000F3: 48 3B D7           cmp         rdx,rdi
  00000000000000F6: 77 05              ja          00000000000000FD
  00000000000000F8: 49 3B C1           cmp         rax,r9
  00000000000000FB: 73 72              jae         000000000000016F
  00000000000000FD: 49 8D 0C 8A        lea         rcx,[r10+rcx*4]
  0000000000000101: 48 3B D1           cmp         rdx,rcx
  0000000000000104: 77 05              ja          000000000000010B
  0000000000000106: 49 3B C2           cmp         rax,r10
  0000000000000109: 73 64              jae         000000000000016F
  000000000000010B: 4C 3B CE           cmp         r9,rsi
  000000000000010E: 77 05              ja          0000000000000115
  0000000000000110: 49 3B FC           cmp         rdi,r12
  0000000000000113: 73 5A              jae         000000000000016F
  0000000000000115: 4C 3B C9           cmp         r9,rcx
  0000000000000118: 77 05              ja          000000000000011F
  000000000000011A: 49 3B FA           cmp         rdi,r10
  000000000000011D: 73 50              jae         000000000000016F
  000000000000011F: 41 8B FB           mov         edi,r11d
  0000000000000122: 83 E7 F8           and         edi,0FFFFFFF8h
  0000000000000125: 49 8B F2           mov         rsi,r10
  0000000000000128: 4D 8B F4           mov         r14,r12
  000000000000012B: 49 2B F1           sub         rsi,r9
  000000000000012E: 4D 2B F1           sub         r14,r9
  0000000000000131: 48 8B CA           mov         rcx,rdx
  0000000000000134: 49 8B C1           mov         rax,r9
  0000000000000137: 49 2B C9           sub         rcx,r9
  000000000000013A: 66 0F 1F 44 00 00  nop         word ptr [rax+rax]
  0000000000000140: C5 FC 10 08        vmovups     ymm1,ymmword ptr [rax]
  0000000000000144: C4 E2 65 B8 0C 06  vfmadd231ps ymm1,ymm3,ymmword ptr [rsi+rax]
  000000000000014A: C5 FC 11 08        vmovups     ymmword ptr [rax],ymm1
  000000000000014E: C5 FC 10 0C 01     vmovups     ymm1,ymmword ptr [rcx+rax]
  0000000000000153: C4 C2 65 B8 0C 06  vfmadd231ps ymm1,ymm3,ymmword ptr [r14+rax]
  0000000000000159: 41 83 C0 08        add         r8d,8
  000000000000015D: 48 83 C3 08        add         rbx,8
  0000000000000161: C5 FC 11 0C 01     vmovups     ymmword ptr [rcx+rax],ymm1
  0000000000000166: 48 8D 40 20        lea         rax,[rax+20h]
  000000000000016A: 44 3B C7           cmp         r8d,edi
  000000000000016D: 72 D1              jb          0000000000000140
  000000000000016F: 45 3B C3           cmp         r8d,r11d
  0000000000000172: 0F 83 30 01 00 00  jae         00000000000002A8
  0000000000000178: 41 8B C3           mov         eax,r11d
  000000000000017B: 41 2B C0           sub         eax,r8d
  000000000000017E: 83 F8 04           cmp         eax,4
  0000000000000181: 0F 82 DA 00 00 00  jb          0000000000000261
  0000000000000187: 41 8B C3           mov         eax,r11d
  000000000000018A: 48 8D 4B 01        lea         rcx,[rbx+1]
  000000000000018E: 41 2B C0           sub         eax,r8d
  0000000000000191: 49 8D 0C 89        lea         rcx,[r9+rcx*4]
  0000000000000195: 83 E8 04           sub         eax,4
  0000000000000198: 48 8B FA           mov         rdi,rdx
  000000000000019B: C1 E8 02           shr         eax,2
  000000000000019E: 49 8B F2           mov         rsi,r10
  00000000000001A1: 4D 8B F4           mov         r14,r12
  00000000000001A4: 49 2B F9           sub         rdi,r9
  00000000000001A7: 49 2B F1           sub         rsi,r9
  00000000000001AA: 4D 2B F1           sub         r14,r9
  00000000000001AD: FF C0              inc         eax
  00000000000001AF: 44 8B F8           mov         r15d,eax
  00000000000001B2: 45 8D 04 80        lea         r8d,[r8+rax*4]
  00000000000001B6: 48 8D 1C 83        lea         rbx,[rbx+rax*4]
  00000000000001BA: 66 0F 1F 44 00 00  nop         word ptr [rax+rax]
  00000000000001C0: C5 FA 10 44 31 FC  vmovss      xmm0,dword ptr [rcx+rsi-4]
  00000000000001C6: C4 E2 69 A9 41 FC  vfmadd213ss xmm0,xmm2,dword ptr [rcx-4]
  00000000000001CC: C5 FA 11 41 FC     vmovss      dword ptr [rcx-4],xmm0
  00000000000001D1: C4 C1 7A 10 44 0E  vmovss      xmm0,dword ptr [r14+rcx-4]
                    FC
  00000000000001D8: C4 E2 69 A9 44 0F  vfmadd213ss xmm0,xmm2,dword ptr [rdi+rcx-4]
                    FC
  00000000000001DF: C5 FA 11 44 0F FC  vmovss      dword ptr [rdi+rcx-4],xmm0
  00000000000001E5: C5 FA 10 0C 31     vmovss      xmm1,dword ptr [rcx+rsi]
  00000000000001EA: C4 E2 69 A9 09     vfmadd213ss xmm1,xmm2,dword ptr [rcx]
  00000000000001EF: C5 FA 11 09        vmovss      dword ptr [rcx],xmm1
  00000000000001F3: C4 C1 7A 10 04 0E  vmovss      xmm0,dword ptr [r14+rcx]
  00000000000001F9: C4 E2 69 A9 04 0F  vfmadd213ss xmm0,xmm2,dword ptr [rdi+rcx]
  00000000000001FF: C5 FA 11 04 0F     vmovss      dword ptr [rdi+rcx],xmm0
  0000000000000204: C5 FA 10 4C 31 04  vmovss      xmm1,dword ptr [rcx+rsi+4]
  000000000000020A: C4 E2 69 A9 49 04  vfmadd213ss xmm1,xmm2,dword ptr [rcx+4]
  0000000000000210: C5 FA 11 49 04     vmovss      dword ptr [rcx+4],xmm1
  0000000000000215: C4 C1 7A 10 44 0E  vmovss      xmm0,dword ptr [r14+rcx+4]
                    04
  000000000000021C: C4 E2 69 A9 44 0F  vfmadd213ss xmm0,xmm2,dword ptr [rdi+rcx+4]
                    04
  0000000000000223: C5 FA 11 44 0F 04  vmovss      dword ptr [rdi+rcx+4],xmm0
  0000000000000229: C5 FA 10 4C 31 08  vmovss      xmm1,dword ptr [rcx+rsi+8]
  000000000000022F: C4 E2 69 A9 49 08  vfmadd213ss xmm1,xmm2,dword ptr [rcx+8]
  0000000000000235: C5 FA 11 49 08     vmovss      dword ptr [rcx+8],xmm1
  000000000000023A: C4 C1 7A 10 44 0E  vmovss      xmm0,dword ptr [r14+rcx+8]
                    08
  0000000000000241: C4 E2 69 A9 44 0F  vfmadd213ss xmm0,xmm2,dword ptr [rdi+rcx+8]
                    08
  0000000000000248: C5 FA 11 44 0F 08  vmovss      dword ptr [rdi+rcx+8],xmm0
  000000000000024E: 48 8D 49 10        lea         rcx,[rcx+10h]
  0000000000000252: 49 83 EF 01        sub         r15,1
  0000000000000256: 0F 85 64 FF FF FF  jne         00000000000001C0
  000000000000025C: 4C 8B 7C 24 08     mov         r15,qword ptr [rsp+8]
  0000000000000261: 45 3B C3           cmp         r8d,r11d
  0000000000000264: 73 42              jae         00000000000002A8
  0000000000000266: 49 8D 0C 99        lea         rcx,[r9+rbx*4]
  000000000000026A: 4D 2B D1           sub         r10,r9
  000000000000026D: 49 8B DC           mov         rbx,r12
  0000000000000270: 49 2B D1           sub         rdx,r9
  0000000000000273: 49 2B D9           sub         rbx,r9
  0000000000000276: 41 8B C3           mov         eax,r11d
  0000000000000279: 41 2B C0           sub         eax,r8d
  000000000000027C: 44 8B C0           mov         r8d,eax
  000000000000027F: C4 A1 7A 10 04 11  vmovss      xmm0,dword ptr [rcx+r10]
  0000000000000285: C4 E2 69 A9 01     vfmadd213ss xmm0,xmm2,dword ptr [rcx]
  000000000000028A: C5 FA 11 01        vmovss      dword ptr [rcx],xmm0
  000000000000028E: C5 FA 10 0C 0B     vmovss      xmm1,dword ptr [rbx+rcx]
  0000000000000293: C4 E2 69 A9 0C 0A  vfmadd213ss xmm1,xmm2,dword ptr [rdx+rcx]
  0000000000000299: C5 FA 11 0C 0A     vmovss      dword ptr [rdx+rcx],xmm1
  000000000000029E: 48 8D 49 04        lea         rcx,[rcx+4]
  00000000000002A2: 49 83 E8 01        sub         r8,1
  00000000000002A6: 75 D7              jne         000000000000027F
  00000000000002A8: 4C 8B 94 24 88 00  mov         r10,qword ptr [rsp+0000000000000088h]
                    00 00
  00000000000002B0: 4D 85 D2           test        r10,r10
  00000000000002B3: 74 0A              je          00000000000002BF
  00000000000002B5: C4 C1 6A 58 07     vaddss      xmm0,xmm2,dword ptr [r15]
  00000000000002BA: C4 C1 7A 11 07     vmovss      dword ptr [r15],xmm0
  00000000000002BF: 8B 94 24 A0 00 00  mov         edx,dword ptr [rsp+00000000000000A0h]
                    00
  00000000000002C6: 49 83 C7 04        add         r15,4
  00000000000002CA: 8B 44 24 04        mov         eax,dword ptr [rsp+4]
  00000000000002CE: 41 FF C5           inc         r13d
  00000000000002D1: 48 8B 5C 24 58     mov         rbx,qword ptr [rsp+58h]
  00000000000002D6: 4C 89 7C 24 08     mov         qword ptr [rsp+8],r15
  00000000000002DB: 44 3B EA           cmp         r13d,edx
  00000000000002DE: 0F 82 AC FD FF FF  jb          0000000000000090
  00000000000002E4: 44 8B 04 24        mov         r8d,dword ptr [rsp]
  00000000000002E8: 48 8B 7C 24 60     mov         rdi,qword ptr [rsp+60h]
  00000000000002ED: 4C 8B 4C 24 68     mov         r9,qword ptr [rsp+68h]
  00000000000002F2: 41 FF C0           inc         r8d
  00000000000002F5: 44 89 04 24        mov         dword ptr [rsp],r8d
  00000000000002F9: 44 3B 84 24 90 00  cmp         r8d,dword ptr [rsp+0000000000000090h]
                    00 00
  0000000000000301: 0F 82 59 FD FF FF  jb          0000000000000060
  0000000000000307: 4C 8B 7C 24 10     mov         r15,qword ptr [rsp+10h]
  000000000000030C: 4C 8B 74 24 18     mov         r14,qword ptr [rsp+18h]
  0000000000000311: 4C 8B 6C 24 20     mov         r13,qword ptr [rsp+20h]
  0000000000000316: 4C 8B 64 24 28     mov         r12,qword ptr [rsp+28h]
  000000000000031B: 48 8B 74 24 30     mov         rsi,qword ptr [rsp+30h]
  0000000000000320: C5 F8 77           vzeroupper
  0000000000000323: 48 83 C4 38        add         rsp,38h
  0000000000000327: 5F                 pop         rdi
  0000000000000328: 5B                 pop         rbx
  0000000000000329: C3                 ret

matmul_forward:
  0000000000000000: 48 8B C4           mov         rax,rsp
  0000000000000003: 4C 89 48 20        mov         qword ptr [rax+20h],r9
  0000000000000007: 4C 89 40 18        mov         qword ptr [rax+18h],r8
  000000000000000B: 48 89 50 10        mov         qword ptr [rax+10h],rdx
  000000000000000F: 48 89 48 08        mov         qword ptr [rax+8],rcx
  0000000000000013: 41 55              push        r13
  0000000000000015: 48 83 EC 40        sub         rsp,40h
  0000000000000019: 45 33 ED           xor         r13d,r13d
  000000000000001C: 44 89 2C 24        mov         dword ptr [rsp],r13d
  0000000000000020: 44 39 6C 24 70     cmp         dword ptr [rsp+70h],r13d
  0000000000000025: 0F 86 DA 01 00 00  jbe         0000000000000205
  000000000000002B: 44 8B 54 24 78     mov         r10d,dword ptr [rsp+78h]
  0000000000000030: 48 89 58 F0        mov         qword ptr [rax-10h],rbx
  0000000000000034: 48 89 70 E8        mov         qword ptr [rax-18h],rsi
  0000000000000038: 48 89 78 E0        mov         qword ptr [rax-20h],rdi
  000000000000003C: 4C 89 60 D8        mov         qword ptr [rax-28h],r12
  0000000000000040: 44 8B A4 24 80 00  mov         r12d,dword ptr [rsp+0000000000000080h]
                    00 00
  0000000000000048: 4C 89 70 D0        mov         qword ptr [rax-30h],r14
  000000000000004C: 4C 89 78 C8        mov         qword ptr [rax-38h],r15
  0000000000000050: 41 8B C5           mov         eax,r13d
  0000000000000053: 45 33 F6           xor         r14d,r14d
  0000000000000056: 41 0F AF C2        imul        eax,r10d
  000000000000005A: 48 8D 34 82        lea         rsi,[rdx+rax*4]
  000000000000005E: 41 8B C5           mov         eax,r13d
  0000000000000061: 41 0F AF C4        imul        eax,r12d
  0000000000000065: 48 8D 3C 81        lea         rdi,[rcx+rax*4]
  0000000000000069: 45 85 E4           test        r12d,r12d
  000000000000006C: 0F 84 63 01 00 00  je          00000000000001D5
  0000000000000072: 4C 8B 6C 24 68     mov         r13,qword ptr [rsp+68h]
  0000000000000077: 4D 8B FD           mov         r15,r13
  000000000000007A: 4C 2B FF           sub         r15,rdi
  000000000000007D: 0F 1F 00           nop         dword ptr [rax]
  0000000000000080: 4D 85 ED           test        r13,r13
  0000000000000083: 74 08              je          000000000000008D
  0000000000000085: C4 A1 7A 10 1C 3F  vmovss      xmm3,dword ptr [rdi+r15]
  000000000000008B: EB 04              jmp         0000000000000091
  000000000000008D: C5 E0 57 DB        vxorps      xmm3,xmm3,xmm3
  0000000000000091: 41 8B C6           mov         eax,r14d
  0000000000000094: 33 DB              xor         ebx,ebx
  0000000000000096: 41 0F AF C2        imul        eax,r10d
  000000000000009A: 4D 8D 0C 80        lea         r9,[r8+rax*4]
  000000000000009E: 45 33 C0           xor         r8d,r8d
  00000000000000A1: 45 85 D2           test        r10d,r10d
  00000000000000A4: 0F 84 04 01 00 00  je          00000000000001AE
  00000000000000AA: 41 83 FA 10        cmp         r10d,10h
  00000000000000AE: 72 61              jb          0000000000000111
  00000000000000B0: 41 8B D2           mov         edx,r10d
  00000000000000B3: 48 8D 46 20        lea         rax,[rsi+20h]
  00000000000000B7: 83 E2 F0           and         edx,0FFFFFFF0h
  00000000000000BA: 49 8B C9           mov         rcx,r9
  00000000000000BD: 48 2B CE           sub         rcx,rsi
  00000000000000C0: C5 E8 57 D2        vxorps      xmm2,xmm2,xmm2
  00000000000000C4: C5 D8 57 E4        vxorps      xmm4,xmm4,xmm4
  00000000000000C8: 0F 1F 84 00 00 00  nop         dword ptr [rax+rax+0000000000000000h]
                    00 00
  00000000000000D0: C5 FC 10 48 E0     vmovups     ymm1,ymmword ptr [rax-20h]
  00000000000000D5: C4 E2 75 B8 54 08  vfmadd231ps ymm2,ymm1,ymmword ptr [rax+rcx-20h]
                    E0
  00000000000000DC: C5 FC 10 08        vmovups     ymm1,ymmword ptr [rax]
  00000000000000E0: C4 E2 75 B8 24 08  vfmadd231ps ymm4,ymm1,ymmword ptr [rax+rcx]
  00000000000000E6: 41 83 C0 10        add         r8d,10h
  00000000000000EA: 48 8D 40 40        lea         rax,[rax+40h]
  00000000000000EE: 48 83 C3 10        add         rbx,10h
  00000000000000F2: 44 3B C2           cmp         r8d,edx
  00000000000000F5: 72 D9              jb          00000000000000D0
  00000000000000F7: C5 DC 58 C2        vaddps      ymm0,ymm4,ymm2
  00000000000000FB: C5 FF 7C C8        vhaddps     ymm1,ymm0,ymm0
  00000000000000FF: C5 F7 7C D1        vhaddps     ymm2,ymm1,ymm1
  0000000000000103: C4 E3 7D 19 D0 01  vextractf128 xmm0,ymm2,1
  0000000000000109: C5 F8 58 C2        vaddps      xmm0,xmm0,xmm2
  000000000000010D: C5 E2 58 D8        vaddss      xmm3,xmm3,xmm0
  0000000000000111: 45 3B C2           cmp         r8d,r10d
  0000000000000114: 0F 83 94 00 00 00  jae         00000000000001AE
  000000000000011A: 41 8B C2           mov         eax,r10d
  000000000000011D: 41 2B C0           sub         eax,r8d
  0000000000000120: 83 F8 04           cmp         eax,4
  0000000000000123: 72 68              jb          000000000000018D
  0000000000000125: 41 8B C2           mov         eax,r10d
  0000000000000128: 48 8D 4B 01        lea         rcx,[rbx+1]
  000000000000012C: 41 2B C0           sub         eax,r8d
  000000000000012F: 48 8D 0C 8E        lea         rcx,[rsi+rcx*4]
  0000000000000133: 83 E8 04           sub         eax,4
  0000000000000136: 49 8B D1           mov         rdx,r9
  0000000000000139: C1 E8 02           shr         eax,2
  000000000000013C: 48 2B D6           sub         rdx,rsi
  000000000000013F: FF C0              inc         eax
  0000000000000141: 44 8B D8           mov         r11d,eax
  0000000000000144: 45 8D 04 80        lea         r8d,[r8+rax*4]
  0000000000000148: 48 8D 1C 83        lea         rbx,[rbx+rax*4]
  000000000000014C: 0F 1F 40 00        nop         dword ptr [rax]
  0000000000000150: C5 FA 10 41 FC     vmovss      xmm0,dword ptr [rcx-4]
  0000000000000155: C4 E2 79 B9 5C 11  vfmadd231ss xmm3,xmm0,dword ptr [rcx+rdx-4]
                    FC
  000000000000015C: C5 FA 10 01        vmovss      xmm0,dword ptr [rcx]
  0000000000000160: C4 E2 79 B9 1C 11  vfmadd231ss xmm3,xmm0,dword ptr [rcx+rdx]
  0000000000000166: C5 FA 10 49 04     vmovss      xmm1,dword ptr [rcx+4]
  000000000000016B: C4 E2 71 B9 5C 11  vfmadd231ss xmm3,xmm1,dword ptr [rcx+rdx+4]
                    04
  0000000000000172: C5 FA 10 41 08     vmovss      xmm0,dword ptr [rcx+8]
  0000000000000177: C4 E2 79 B9 5C 11  vfmadd231ss xmm3,xmm0,dword ptr [rcx+rdx+8]
                    08
  000000000000017E: 48 8D 49 10        lea         rcx,[rcx+10h]
  0000000000000182: 49 83 EB 01        sub         r11,1
  0000000000000186: 75 C8              jne         0000000000000150
  0000000000000188: 45 3B C2           cmp         r8d,r10d
  000000000000018B: 73 21              jae         00000000000001AE
  000000000000018D: 4C 2B CE           sub         r9,rsi
  0000000000000190: 48 8D 0C 9E        lea         rcx,[rsi+rbx*4]
  0000000000000194: 41 8B D2           mov         edx,r10d
  0000000000000197: 41 2B D0           sub         edx,r8d
  000000000000019A: C5 FA 10 01        vmovss      xmm0,dword ptr [rcx]
  000000000000019E: C4 C2 79 B9 1C 09  vfmadd231ss xmm3,xmm0,dword ptr [r9+rcx]
  00000000000001A4: 48 8D 49 04        lea         rcx,[rcx+4]
  00000000000001A8: 48 83 EA 01        sub         rdx,1
  00000000000001AC: 75 EC              jne         000000000000019A
  00000000000001AE: 4C 8B 44 24 60     mov         r8,qword ptr [rsp+60h]
  00000000000001B3: 41 FF C6           inc         r14d
  00000000000001B6: C5 FA 11 1F        vmovss      dword ptr [rdi],xmm3
  00000000000001BA: 48 83 C7 04        add         rdi,4
  00000000000001BE: 45 3B F4           cmp         r14d,r12d
  00000000000001C1: 0F 82 B9 FE FF FF  jb          0000000000000080
  00000000000001C7: 44 8B 2C 24        mov         r13d,dword ptr [rsp]
  00000000000001CB: 48 8B 54 24 58     mov         rdx,qword ptr [rsp+58h]
  00000000000001D0: 48 8B 4C 24 50     mov         rcx,qword ptr [rsp+50h]
  00000000000001D5: 41 FF C5           inc         r13d
  00000000000001D8: 44 89 2C 24        mov         dword ptr [rsp],r13d
  00000000000001DC: 44 3B 6C 24 70     cmp         r13d,dword ptr [rsp+70h]
  00000000000001E1: 0F 82 69 FE FF FF  jb          0000000000000050
  00000000000001E7: 4C 8B 7C 24 10     mov         r15,qword ptr [rsp+10h]
  00000000000001EC: 4C 8B 74 24 18     mov         r14,qword ptr [rsp+18h]
  00000000000001F1: 4C 8B 64 24 20     mov         r12,qword ptr [rsp+20h]
  00000000000001F6: 48 8B 7C 24 28     mov         rdi,qword ptr [rsp+28h]
  00000000000001FB: 48 8B 74 24 30     mov         rsi,qword ptr [rsp+30h]
  0000000000000200: 48 8B 5C 24 38     mov         rbx,qword ptr [rsp+38h]
  0000000000000205: C5 F8 77           vzeroupper
  0000000000000208: 48 83 C4 40        add         rsp,40h
  000000000000020C: 41 5D              pop         r13
  000000000000020E: C3                 ret

memcpy_0:
  0000000000000000: 48 89 5C 24 08     mov         qword ptr [rsp+8],rbx
  0000000000000005: 45 33 D2           xor         r10d,r10d
  0000000000000008: 4C 8B D9           mov         r11,rcx
  000000000000000B: 41 83 F8 04        cmp         r8d,4
  000000000000000F: 72 66              jb          0000000000000077
  0000000000000011: 41 8D 40 FC        lea         eax,[r8-4]
  0000000000000015: 48 2B D1           sub         rdx,rcx
  0000000000000018: C1 E8 02           shr         eax,2
  000000000000001B: 4C 8D 49 04        lea         r9,[rcx+4]
  000000000000001F: FF C0              inc         eax
  0000000000000021: 8B C8              mov         ecx,eax
  0000000000000023: 44 8D 14 85 00 00  lea         r10d,[rax*4+0000000000000000h]
                    00 00
  000000000000002B: 48 8D 1C 85 00 00  lea         rbx,[rax*4+0000000000000000h]
                    00 00
  0000000000000033: 0F 1F 40 00        nop         dword ptr [rax]
  0000000000000037: 66 0F 1F 84 00 00  nop         word ptr [rax+rax+0000000000000000h]
                    00 00 00
  0000000000000040: 41 8B 44 11 FC     mov         eax,dword ptr [r9+rdx-4]
  0000000000000045: 41 89 41 FC        mov         dword ptr [r9-4],eax
  0000000000000049: 41 8B 04 11        mov         eax,dword ptr [r9+rdx]
  000000000000004D: 41 89 01           mov         dword ptr [r9],eax
  0000000000000050: 41 8B 44 11 04     mov         eax,dword ptr [r9+rdx+4]
  0000000000000055: 41 89 41 04        mov         dword ptr [r9+4],eax
  0000000000000059: 41 8B 44 11 08     mov         eax,dword ptr [r9+rdx+8]
  000000000000005E: 41 89 41 08        mov         dword ptr [r9+8],eax
  0000000000000062: 4D 8D 49 10        lea         r9,[r9+10h]
  0000000000000066: 48 83 E9 01        sub         rcx,1
  000000000000006A: 75 D4              jne         0000000000000040
  000000000000006C: 45 3B D0           cmp         r10d,r8d
  000000000000006F: 72 11              jb          0000000000000082
  0000000000000071: 48 8B 5C 24 08     mov         rbx,qword ptr [rsp+8]
  0000000000000076: C3                 ret
  0000000000000077: 45 85 C0           test        r8d,r8d
  000000000000007A: 74 1C              je          0000000000000098
  000000000000007C: 49 8B DA           mov         rbx,r10
  000000000000007F: 49 2B D3           sub         rdx,r11
  0000000000000082: 49 8D 0C 9B        lea         rcx,[r11+rbx*4]
  0000000000000086: 45 2B C2           sub         r8d,r10d
  0000000000000089: 8B 04 0A           mov         eax,dword ptr [rdx+rcx]
  000000000000008C: 89 01              mov         dword ptr [rcx],eax
  000000000000008E: 48 8D 49 04        lea         rcx,[rcx+4]
  0000000000000092: 49 83 E8 01        sub         r8,1
  0000000000000096: 75 F1              jne         0000000000000089
  0000000000000098: 48 8B 5C 24 08     mov         rbx,qword ptr [rsp+8]
  000000000000009D: C3                 ret

  Summary

          A0 .chks64
          70 .debug$S
          2F .drectve
          54 .pdata
         5D7 .text$mn
          88 .xdata
