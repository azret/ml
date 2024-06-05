Microsoft (R) COFF/PE Dumper Version 14.39.33523.0
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file MatMul.obj

File Type: COFF OBJECT

matmul_backward:
  0000000000000000: 48 8B C4           mov         rax,rsp
  0000000000000003: 4C 89 48 20        mov         qword ptr [rax+20h],r9
  0000000000000007: 4C 89 40 18        mov         qword ptr [rax+18h],r8
  000000000000000B: 48 89 50 10        mov         qword ptr [rax+10h],rdx
  000000000000000F: 48 83 EC 48        sub         rsp,48h
  0000000000000013: 4C 8B D2           mov         r10,rdx
  0000000000000016: 4D 8B D8           mov         r11,r8
  0000000000000019: 33 D2              xor         edx,edx
  000000000000001B: 89 14 24           mov         dword ptr [rsp],edx
  000000000000001E: 39 94 24 90 00 00  cmp         dword ptr [rsp+0000000000000090h],edx
                    00
  0000000000000025: 0F 86 EA 02 00 00  jbe         0000000000000315
  000000000000002B: 4C 8B 84 24 88 00  mov         r8,qword ptr [rsp+0000000000000088h]
                    00 00
  0000000000000033: 48 89 58 08        mov         qword ptr [rax+8],rbx
  0000000000000037: 8B 9C 24 98 00 00  mov         ebx,dword ptr [rsp+0000000000000098h]
                    00
  000000000000003E: 48 89 68 F8        mov         qword ptr [rax-8],rbp
  0000000000000042: 48 89 70 F0        mov         qword ptr [rax-10h],rsi
  0000000000000046: 48 89 78 E8        mov         qword ptr [rax-18h],rdi
  000000000000004A: 4C 89 60 E0        mov         qword ptr [rax-20h],r12
  000000000000004E: 44 8B A4 24 A0 00  mov         r12d,dword ptr [rsp+00000000000000A0h]
                    00 00
  0000000000000056: 4C 89 68 D8        mov         qword ptr [rax-28h],r13
  000000000000005A: 4C 89 70 D0        mov         qword ptr [rax-30h],r14
  000000000000005E: 4C 89 78 C8        mov         qword ptr [rax-38h],r15
  0000000000000062: 0F 1F 40 00        nop         dword ptr [rax]
  0000000000000066: 66 66 0F 1F 84 00  nop         word ptr [rax+rax+0000000000000000h]
                    00 00 00 00
  0000000000000070: 8B C2              mov         eax,edx
  0000000000000072: 45 33 FF           xor         r15d,r15d
  0000000000000075: 0F AF C3           imul        eax,ebx
  0000000000000078: 4D 8D 34 83        lea         r14,[r11+rax*4]
  000000000000007C: 4D 8D 0C 81        lea         r9,[r9+rax*4]
  0000000000000080: 45 85 E4           test        r12d,r12d
  0000000000000083: 0F 84 4D 02 00 00  je          00000000000002D6
  0000000000000089: 8B C2              mov         eax,edx
  000000000000008B: 4D 8B E8           mov         r13,r8
  000000000000008E: 41 0F AF C4        imul        eax,r12d
  0000000000000092: 89 44 24 04        mov         dword ptr [rsp+4],eax
  0000000000000096: 66 66 0F 1F 84 00  nop         word ptr [rax+rax+0000000000000000h]
                    00 00 00 00
  00000000000000A0: 41 03 C7           add         eax,r15d
  00000000000000A3: 41 8B CF           mov         ecx,r15d
  00000000000000A6: 0F AF CB           imul        ecx,ebx
  00000000000000A9: 33 D2              xor         edx,edx
  00000000000000AB: 45 33 DB           xor         r11d,r11d
  00000000000000AE: C4 C1 7A 10 14 82  vmovss      xmm2,dword ptr [r10+rax*4]
  00000000000000B4: 48 8B 44 24 70     mov         rax,qword ptr [rsp+70h]
  00000000000000B9: C5 F8 28 DA        vmovaps     xmm3,xmm2
  00000000000000BD: C5 E8 C6 DA 00     vshufps     xmm3,xmm2,xmm2,0
  00000000000000C2: 4C 8D 14 88        lea         r10,[rax+rcx*4]
  00000000000000C6: 48 8B 44 24 78     mov         rax,qword ptr [rsp+78h]
  00000000000000CB: 4C 8D 04 88        lea         r8,[rax+rcx*4]
  00000000000000CF: 85 DB              test        ebx,ebx
  00000000000000D1: 0F 84 C5 01 00 00  je          000000000000029C
  00000000000000D7: 83 FB 04           cmp         ebx,4
  00000000000000DA: 0F 82 8D 00 00 00  jb          000000000000016D
  00000000000000E0: 8D 43 FF           lea         eax,[rbx-1]
  00000000000000E3: 48 63 C8           movsxd      rcx,eax
  00000000000000E6: 49 8D 34 8E        lea         rsi,[r14+rcx*4]
  00000000000000EA: 49 8D 04 88        lea         rax,[r8+rcx*4]
  00000000000000EE: 4C 3B C6           cmp         r8,rsi
  00000000000000F1: 77 05              ja          00000000000000F8
  00000000000000F3: 49 3B C6           cmp         rax,r14
  00000000000000F6: 73 75              jae         000000000000016D
  00000000000000F8: 49 8D 3C 89        lea         rdi,[r9+rcx*4]
  00000000000000FC: 4C 3B C7           cmp         r8,rdi
  00000000000000FF: 77 05              ja          0000000000000106
  0000000000000101: 49 3B C1           cmp         rax,r9
  0000000000000104: 73 67              jae         000000000000016D
  0000000000000106: 49 8D 0C 8A        lea         rcx,[r10+rcx*4]
  000000000000010A: 4C 3B C1           cmp         r8,rcx
  000000000000010D: 77 05              ja          0000000000000114
  000000000000010F: 49 3B C2           cmp         rax,r10
  0000000000000112: 73 59              jae         000000000000016D
  0000000000000114: 4C 3B CE           cmp         r9,rsi
  0000000000000117: 77 05              ja          000000000000011E
  0000000000000119: 49 3B FE           cmp         rdi,r14
  000000000000011C: 73 4F              jae         000000000000016D
  000000000000011E: 4C 3B C9           cmp         r9,rcx
  0000000000000121: 77 05              ja          0000000000000128
  0000000000000123: 49 3B FA           cmp         rdi,r10
  0000000000000126: 73 45              jae         000000000000016D
  0000000000000128: 8B FB              mov         edi,ebx
  000000000000012A: 83 E7 FC           and         edi,0FFFFFFFCh
  000000000000012D: 49 8B F2           mov         rsi,r10
  0000000000000130: 49 8B EE           mov         rbp,r14
  0000000000000133: 49 2B F1           sub         rsi,r9
  0000000000000136: 49 2B E9           sub         rbp,r9
  0000000000000139: 49 8B C8           mov         rcx,r8
  000000000000013C: 49 8B C1           mov         rax,r9
  000000000000013F: 49 2B C9           sub         rcx,r9
  0000000000000142: C5 E0 59 0C 06     vmulps      xmm1,xmm3,xmmword ptr [rsi+rax]
  0000000000000147: C5 F0 58 08        vaddps      xmm1,xmm1,xmmword ptr [rax]
  000000000000014B: C5 F8 11 08        vmovups     xmmword ptr [rax],xmm1
  000000000000014F: C5 E0 59 0C 28     vmulps      xmm1,xmm3,xmmword ptr [rax+rbp]
  0000000000000154: C5 F0 58 0C 01     vaddps      xmm1,xmm1,xmmword ptr [rcx+rax]
  0000000000000159: 83 C2 04           add         edx,4
  000000000000015C: 49 83 C3 04        add         r11,4
  0000000000000160: C5 F8 11 0C 01     vmovups     xmmword ptr [rcx+rax],xmm1
  0000000000000165: 48 8D 40 10        lea         rax,[rax+10h]
  0000000000000169: 3B D7              cmp         edx,edi
  000000000000016B: 72 D5              jb          0000000000000142
  000000000000016D: 3B D3              cmp         edx,ebx
  000000000000016F: 0F 83 27 01 00 00  jae         000000000000029C
  0000000000000175: 8B C3              mov         eax,ebx
  0000000000000177: 2B C2              sub         eax,edx
  0000000000000179: 83 F8 04           cmp         eax,4
  000000000000017C: 0F 82 D6 00 00 00  jb          0000000000000258
  0000000000000182: 8B C3              mov         eax,ebx
  0000000000000184: 49 8D 4B 01        lea         rcx,[r11+1]
  0000000000000188: 2B C2              sub         eax,edx
  000000000000018A: 49 8D 0C 89        lea         rcx,[r9+rcx*4]
  000000000000018E: 83 E8 04           sub         eax,4
  0000000000000191: 49 8B F8           mov         rdi,r8
  0000000000000194: C1 E8 02           shr         eax,2
  0000000000000197: 49 8B F2           mov         rsi,r10
  000000000000019A: 49 8B EE           mov         rbp,r14
  000000000000019D: 49 2B F9           sub         rdi,r9
  00000000000001A0: 49 2B F1           sub         rsi,r9
  00000000000001A3: 49 2B E9           sub         rbp,r9
  00000000000001A6: FF C0              inc         eax
  00000000000001A8: 44 8B E0           mov         r12d,eax
  00000000000001AB: 8D 14 82           lea         edx,[rdx+rax*4]
  00000000000001AE: 4D 8D 1C 83        lea         r11,[r11+rax*4]
  00000000000001B2: 0F 1F 40 00        nop         dword ptr [rax]
  00000000000001B6: 66 66 0F 1F 84 00  nop         word ptr [rax+rax+0000000000000000h]
                    00 00 00 00
  00000000000001C0: C5 EA 59 44 31 FC  vmulss      xmm0,xmm2,dword ptr [rcx+rsi-4]
  00000000000001C6: C5 FA 58 49 FC     vaddss      xmm1,xmm0,dword ptr [rcx-4]
  00000000000001CB: C5 FA 11 49 FC     vmovss      dword ptr [rcx-4],xmm1
  00000000000001D0: C5 EA 59 44 29 FC  vmulss      xmm0,xmm2,dword ptr [rcx+rbp-4]
  00000000000001D6: C5 FA 58 4C 0F FC  vaddss      xmm1,xmm0,dword ptr [rdi+rcx-4]
  00000000000001DC: C5 FA 11 4C 0F FC  vmovss      dword ptr [rdi+rcx-4],xmm1
  00000000000001E2: C5 EA 59 04 31     vmulss      xmm0,xmm2,dword ptr [rcx+rsi]
  00000000000001E7: C5 FA 58 09        vaddss      xmm1,xmm0,dword ptr [rcx]
  00000000000001EB: C5 FA 11 09        vmovss      dword ptr [rcx],xmm1
  00000000000001EF: C5 EA 59 04 29     vmulss      xmm0,xmm2,dword ptr [rcx+rbp]
  00000000000001F4: C5 FA 58 0C 0F     vaddss      xmm1,xmm0,dword ptr [rdi+rcx]
  00000000000001F9: C5 FA 11 0C 0F     vmovss      dword ptr [rdi+rcx],xmm1
  00000000000001FE: C5 EA 59 44 31 04  vmulss      xmm0,xmm2,dword ptr [rcx+rsi+4]
  0000000000000204: C5 FA 58 49 04     vaddss      xmm1,xmm0,dword ptr [rcx+4]
  0000000000000209: C5 FA 11 49 04     vmovss      dword ptr [rcx+4],xmm1
  000000000000020E: C5 EA 59 44 29 04  vmulss      xmm0,xmm2,dword ptr [rcx+rbp+4]
  0000000000000214: C5 FA 58 4C 0F 04  vaddss      xmm1,xmm0,dword ptr [rdi+rcx+4]
  000000000000021A: C5 FA 11 4C 0F 04  vmovss      dword ptr [rdi+rcx+4],xmm1
  0000000000000220: C5 EA 59 44 31 08  vmulss      xmm0,xmm2,dword ptr [rcx+rsi+8]
  0000000000000226: C5 FA 58 49 08     vaddss      xmm1,xmm0,dword ptr [rcx+8]
  000000000000022B: C5 FA 11 49 08     vmovss      dword ptr [rcx+8],xmm1
  0000000000000230: C5 EA 59 44 29 08  vmulss      xmm0,xmm2,dword ptr [rcx+rbp+8]
  0000000000000236: C5 FA 58 4C 0F 08  vaddss      xmm1,xmm0,dword ptr [rdi+rcx+8]
  000000000000023C: C5 FA 11 4C 0F 08  vmovss      dword ptr [rdi+rcx+8],xmm1
  0000000000000242: 48 8D 49 10        lea         rcx,[rcx+10h]
  0000000000000246: 49 83 EC 01        sub         r12,1
  000000000000024A: 0F 85 70 FF FF FF  jne         00000000000001C0
  0000000000000250: 44 8B A4 24 A0 00  mov         r12d,dword ptr [rsp+00000000000000A0h]
                    00 00
  0000000000000258: 3B D3              cmp         edx,ebx
  000000000000025A: 73 40              jae         000000000000029C
  000000000000025C: 4B 8D 0C 99        lea         rcx,[r9+r11*4]
  0000000000000260: 4D 2B D1           sub         r10,r9
  0000000000000263: 4D 8B DE           mov         r11,r14
  0000000000000266: 4D 2B C1           sub         r8,r9
  0000000000000269: 4D 2B D9           sub         r11,r9
  000000000000026C: 8B C3              mov         eax,ebx
  000000000000026E: 2B C2              sub         eax,edx
  0000000000000270: 8B D0              mov         edx,eax
  0000000000000272: C4 A1 6A 59 04 11  vmulss      xmm0,xmm2,dword ptr [rcx+r10]
  0000000000000278: C5 FA 58 09        vaddss      xmm1,xmm0,dword ptr [rcx]
  000000000000027C: C5 FA 11 09        vmovss      dword ptr [rcx],xmm1
  0000000000000280: C4 C1 6A 59 04 0B  vmulss      xmm0,xmm2,dword ptr [r11+rcx]
  0000000000000286: C4 C1 7A 58 0C 08  vaddss      xmm1,xmm0,dword ptr [r8+rcx]
  000000000000028C: C4 C1 7A 11 0C 08  vmovss      dword ptr [r8+rcx],xmm1
  0000000000000292: 48 8D 49 04        lea         rcx,[rcx+4]
  0000000000000296: 48 83 EA 01        sub         rdx,1
  000000000000029A: 75 D6              jne         0000000000000272
  000000000000029C: 4C 8B 84 24 88 00  mov         r8,qword ptr [rsp+0000000000000088h]
                    00 00
  00000000000002A4: 4D 85 C0           test        r8,r8
  00000000000002A7: 74 0C              je          00000000000002B5
  00000000000002A9: C4 C1 6A 58 45 00  vaddss      xmm0,xmm2,dword ptr [r13]
  00000000000002AF: C4 C1 7A 11 45 00  vmovss      dword ptr [r13],xmm0
  00000000000002B5: 8B 44 24 04        mov         eax,dword ptr [rsp+4]
  00000000000002B9: 41 FF C7           inc         r15d
  00000000000002BC: 4C 8B 54 24 58     mov         r10,qword ptr [rsp+58h]
  00000000000002C1: 49 83 C5 04        add         r13,4
  00000000000002C5: 45 3B FC           cmp         r15d,r12d
  00000000000002C8: 0F 82 D2 FD FF FF  jb          00000000000000A0
  00000000000002CE: 8B 14 24           mov         edx,dword ptr [rsp]
  00000000000002D1: 4C 8B 5C 24 60     mov         r11,qword ptr [rsp+60h]
  00000000000002D6: 4C 8B 4C 24 68     mov         r9,qword ptr [rsp+68h]
  00000000000002DB: FF C2              inc         edx
  00000000000002DD: 89 14 24           mov         dword ptr [rsp],edx
  00000000000002E0: 3B 94 24 90 00 00  cmp         edx,dword ptr [rsp+0000000000000090h]
                    00
  00000000000002E7: 0F 82 83 FD FF FF  jb          0000000000000070
  00000000000002ED: 4C 8B 7C 24 10     mov         r15,qword ptr [rsp+10h]
  00000000000002F2: 4C 8B 74 24 18     mov         r14,qword ptr [rsp+18h]
  00000000000002F7: 4C 8B 6C 24 20     mov         r13,qword ptr [rsp+20h]
  00000000000002FC: 4C 8B 64 24 28     mov         r12,qword ptr [rsp+28h]
  0000000000000301: 48 8B 7C 24 30     mov         rdi,qword ptr [rsp+30h]
  0000000000000306: 48 8B 74 24 38     mov         rsi,qword ptr [rsp+38h]
  000000000000030B: 48 8B 6C 24 40     mov         rbp,qword ptr [rsp+40h]
  0000000000000310: 48 8B 5C 24 50     mov         rbx,qword ptr [rsp+50h]
  0000000000000315: 48 83 C4 48        add         rsp,48h
  0000000000000319: C3                 ret

matmul_forward_kernel:
  0000000000000000: 48 89 74 24 18     mov         qword ptr [rsp+18h],rsi
  0000000000000005: 48 89 7C 24 20     mov         qword ptr [rsp+20h],rdi
  000000000000000A: 41 54              push        r12
  000000000000000C: 41 55              push        r13
  000000000000000E: 41 56              push        r14
  0000000000000010: 41 57              push        r15
  0000000000000012: 44 8B 54 24 48     mov         r10d,dword ptr [rsp+48h]
  0000000000000017: 33 F6              xor         esi,esi
  0000000000000019: 44 8B 7C 24 60     mov         r15d,dword ptr [rsp+60h]
  000000000000001E: 41 8B C2           mov         eax,r10d
  0000000000000021: 44 8B 5C 24 58     mov         r11d,dword ptr [rsp+58h]
  0000000000000026: 4D 8B E1           mov         r12,r9
  0000000000000029: 41 0F AF C3        imul        eax,r11d
  000000000000002D: 4D 8B E8           mov         r13,r8
  0000000000000030: 45 0F AF D7        imul        r10d,r15d
  0000000000000034: 4C 8D 34 82        lea         r14,[rdx+rax*4]
  0000000000000038: 4A 8D 3C 91        lea         rdi,[rcx+r10*4]
  000000000000003C: 45 85 FF           test        r15d,r15d
  000000000000003F: 0F 84 7B 01 00 00  je          00000000000001C0
  0000000000000045: 48 89 5C 24 28     mov         qword ptr [rsp+28h],rbx
  000000000000004A: 48 89 6C 24 30     mov         qword ptr [rsp+30h],rbp
  000000000000004F: 49 8B E9           mov         rbp,r9
  0000000000000052: 48 2B EF           sub         rbp,rdi
  0000000000000055: 4D 85 E4           test        r12,r12
  0000000000000058: 74 07              je          0000000000000061
  000000000000005A: C5 FA 10 1C 2F     vmovss      xmm3,dword ptr [rdi+rbp]
  000000000000005F: EB 04              jmp         0000000000000065
  0000000000000061: C5 E0 57 DB        vxorps      xmm3,xmm3,xmm3
  0000000000000065: 8B C6              mov         eax,esi
  0000000000000067: 33 D2              xor         edx,edx
  0000000000000069: 41 0F AF C3        imul        eax,r11d
  000000000000006D: 45 33 D2           xor         r10d,r10d
  0000000000000070: 4C 8D 0C 85 00 00  lea         r9,[rax*4+0000000000000000h]
                    00 00
  0000000000000078: 4D 03 CD           add         r9,r13
  000000000000007B: 45 85 DB           test        r11d,r11d
  000000000000007E: 0F 84 1F 01 00 00  je          00000000000001A3
  0000000000000084: 41 83 FB 08        cmp         r11d,8
  0000000000000088: 72 6B              jb          00000000000000F5
  000000000000008A: 45 8B C3           mov         r8d,r11d
  000000000000008D: 49 8D 46 10        lea         rax,[r14+10h]
  0000000000000091: 41 83 E0 F8        and         r8d,0FFFFFFF8h
  0000000000000095: 49 8B C9           mov         rcx,r9
  0000000000000098: 49 2B CE           sub         rcx,r14
  000000000000009B: C5 E8 57 D2        vxorps      xmm2,xmm2,xmm2
  000000000000009F: C5 D8 57 E4        vxorps      xmm4,xmm4,xmm4
  00000000000000A3: 0F 1F 40 00        nop         dword ptr [rax]
  00000000000000A7: 66 0F 1F 84 00 00  nop         word ptr [rax+rax+0000000000000000h]
                    00 00 00
  00000000000000B0: C5 F8 10 4C 01 F0  vmovups     xmm1,xmmword ptr [rcx+rax-10h]
  00000000000000B6: C5 F0 59 48 F0     vmulps      xmm1,xmm1,xmmword ptr [rax-10h]
  00000000000000BB: C5 F0 58 D2        vaddps      xmm2,xmm1,xmm2
  00000000000000BF: C5 F8 10 08        vmovups     xmm1,xmmword ptr [rax]
  00000000000000C3: C5 F0 59 0C 01     vmulps      xmm1,xmm1,xmmword ptr [rcx+rax]
  00000000000000C8: 83 C2 08           add         edx,8
  00000000000000CB: 48 8D 40 20        lea         rax,[rax+20h]
  00000000000000CF: 49 83 C2 08        add         r10,8
  00000000000000D3: C5 F0 58 E4        vaddps      xmm4,xmm1,xmm4
  00000000000000D7: 41 3B D0           cmp         edx,r8d
  00000000000000DA: 72 D4              jb          00000000000000B0
  00000000000000DC: C5 D8 58 CA        vaddps      xmm1,xmm4,xmm2
  00000000000000E0: C5 F0 12 C1        vmovhlps    xmm0,xmm1,xmm1
  00000000000000E4: C5 F8 58 D1        vaddps      xmm2,xmm0,xmm1
  00000000000000E8: C5 E8 C6 C2 F5     vshufps     xmm0,xmm2,xmm2,0F5h
  00000000000000ED: C5 EA 58 D0        vaddss      xmm2,xmm2,xmm0
  00000000000000F1: C5 E2 58 DA        vaddss      xmm3,xmm3,xmm2
  00000000000000F5: 41 3B D3           cmp         edx,r11d
  00000000000000F8: 0F 83 A5 00 00 00  jae         00000000000001A3
  00000000000000FE: 41 8B C3           mov         eax,r11d
  0000000000000101: 2B C2              sub         eax,edx
  0000000000000103: 83 F8 04           cmp         eax,4
  0000000000000106: 72 75              jb          000000000000017D
  0000000000000108: 41 8B C3           mov         eax,r11d
  000000000000010B: 49 8D 4E 04        lea         rcx,[r14+4]
  000000000000010F: 2B C2              sub         eax,edx
  0000000000000111: 4A 8D 0C 91        lea         rcx,[rcx+r10*4]
  0000000000000115: 83 E8 04           sub         eax,4
  0000000000000118: 4D 8B C1           mov         r8,r9
  000000000000011B: C1 E8 02           shr         eax,2
  000000000000011E: 4D 2B C6           sub         r8,r14
  0000000000000121: FF C0              inc         eax
  0000000000000123: 8B D8              mov         ebx,eax
  0000000000000125: 8D 14 82           lea         edx,[rdx+rax*4]
  0000000000000128: 4D 8D 14 82        lea         r10,[r10+rax*4]
  000000000000012C: 0F 1F 40 00        nop         dword ptr [rax]
  0000000000000130: C4 A1 7A 10 44 01  vmovss      xmm0,dword ptr [rcx+r8-4]
                    FC
  0000000000000137: C5 FA 59 49 FC     vmulss      xmm1,xmm0,dword ptr [rcx-4]
  000000000000013C: C4 A1 7A 10 14 01  vmovss      xmm2,dword ptr [rcx+r8]
  0000000000000142: C5 EA 59 01        vmulss      xmm0,xmm2,dword ptr [rcx]
  0000000000000146: 48 8D 49 10        lea         rcx,[rcx+10h]
  000000000000014A: C5 F2 58 DB        vaddss      xmm3,xmm1,xmm3
  000000000000014E: C4 A1 7A 10 4C 01  vmovss      xmm1,dword ptr [rcx+r8-0Ch]
                    F4
  0000000000000155: C5 F2 59 51 F4     vmulss      xmm2,xmm1,dword ptr [rcx-0Ch]
  000000000000015A: C5 E2 58 E0        vaddss      xmm4,xmm3,xmm0
  000000000000015E: C4 A1 7A 10 44 01  vmovss      xmm0,dword ptr [rcx+r8-8]
                    F8
  0000000000000165: C5 FA 59 49 F8     vmulss      xmm1,xmm0,dword ptr [rcx-8]
  000000000000016A: C5 DA 58 DA        vaddss      xmm3,xmm4,xmm2
  000000000000016E: C5 E2 58 D9        vaddss      xmm3,xmm3,xmm1
  0000000000000172: 48 83 EB 01        sub         rbx,1
  0000000000000176: 75 B8              jne         0000000000000130
  0000000000000178: 41 3B D3           cmp         edx,r11d
  000000000000017B: 73 26              jae         00000000000001A3
  000000000000017D: 4D 2B CE           sub         r9,r14
  0000000000000180: 4B 8D 0C 96        lea         rcx,[r14+r10*4]
  0000000000000184: 41 8B C3           mov         eax,r11d
  0000000000000187: 2B C2              sub         eax,edx
  0000000000000189: 8B D0              mov         edx,eax
  000000000000018B: C4 C1 7A 10 04 09  vmovss      xmm0,dword ptr [r9+rcx]
  0000000000000191: C5 FA 59 09        vmulss      xmm1,xmm0,dword ptr [rcx]
  0000000000000195: 48 8D 49 04        lea         rcx,[rcx+4]
  0000000000000199: C5 E2 58 D9        vaddss      xmm3,xmm3,xmm1
  000000000000019D: 48 83 EA 01        sub         rdx,1
  00000000000001A1: 75 E8              jne         000000000000018B
  00000000000001A3: C5 FA 11 1F        vmovss      dword ptr [rdi],xmm3
  00000000000001A7: 48 83 C7 04        add         rdi,4
  00000000000001AB: FF C6              inc         esi
  00000000000001AD: 41 3B F7           cmp         esi,r15d
  00000000000001B0: 0F 82 9F FE FF FF  jb          0000000000000055
  00000000000001B6: 48 8B 6C 24 30     mov         rbp,qword ptr [rsp+30h]
  00000000000001BB: 48 8B 5C 24 28     mov         rbx,qword ptr [rsp+28h]
  00000000000001C0: 48 8B 74 24 38     mov         rsi,qword ptr [rsp+38h]
  00000000000001C5: 48 8B 7C 24 40     mov         rdi,qword ptr [rsp+40h]
  00000000000001CA: 41 5F              pop         r15
  00000000000001CC: 41 5E              pop         r14
  00000000000001CE: 41 5D              pop         r13
  00000000000001D0: 41 5C              pop         r12
  00000000000001D2: C3                 ret

  Summary

          88 .chks64
          70 .debug$S
          2F .drectve
          48 .pdata
         4ED .text$mn
          84 .xdata
