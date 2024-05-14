Microsoft (R) COFF/PE Dumper Version 14.39.33523.0
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file MatMul.obj

File Type: COFF OBJECT

matmul_forward_cpu_c:
  0000000000000000: 4C 89 4C 24 20     mov         qword ptr [rsp+20h],r9
  0000000000000005: 4C 89 44 24 18     mov         qword ptr [rsp+18h],r8
  000000000000000A: 41 56              push        r14
  000000000000000C: 48 83 EC 60        sub         rsp,60h
  0000000000000010: 8B 84 24 90 00 00  mov         eax,dword ptr [rsp+0000000000000090h]
                    00
  0000000000000017: 4C 8B F2           mov         r14,rdx
  000000000000001A: 4C 63 94 24 98 00  movsxd      r10,dword ptr [rsp+0000000000000098h]
                    00 00
  0000000000000022: 4C 8B D9           mov         r11,rcx
  0000000000000025: 85 C0              test        eax,eax
  0000000000000027: 0F 8E 6B 02 00 00  jle         0000000000000298
  000000000000002D: 48 63 94 24 A0 00  movsxd      rdx,dword ptr [rsp+00000000000000A0h]
                    00 00
  0000000000000035: 33 C9              xor         ecx,ecx
  0000000000000037: 48 89 5C 24 70     mov         qword ptr [rsp+70h],rbx
  000000000000003C: 45 33 C0           xor         r8d,r8d
  000000000000003F: 48 89 6C 24 58     mov         qword ptr [rsp+58h],rbp
  0000000000000044: 4C 8B CA           mov         r9,rdx
  0000000000000047: 48 89 74 24 50     mov         qword ptr [rsp+50h],rsi
  000000000000004C: 48 89 7C 24 48     mov         qword ptr [rsp+48h],rdi
  0000000000000051: 4C 89 64 24 40     mov         qword ptr [rsp+40h],r12
  0000000000000056: 45 33 E4           xor         r12d,r12d
  0000000000000059: 49 C1 E1 02        shl         r9,2
  000000000000005D: 4C 89 6C 24 38     mov         qword ptr [rsp+38h],r13
  0000000000000062: 4C 89 4C 24 20     mov         qword ptr [rsp+20h],r9
  0000000000000067: 4C 89 7C 24 30     mov         qword ptr [rsp+30h],r15
  000000000000006C: 89 8C 24 90 00 00  mov         dword ptr [rsp+0000000000000090h],ecx
                    00
  0000000000000073: 4C 89 44 24 10     mov         qword ptr [rsp+10h],r8
  0000000000000078: 48 89 44 24 18     mov         qword ptr [rsp+18h],rax
  000000000000007D: 0F 1F 00           nop         dword ptr [rax]
  0000000000000080: 85 D2              test        edx,edx
  0000000000000082: 0F 8E C9 01 00 00  jle         0000000000000251
  0000000000000088: 48 8B 94 24 88 00  mov         rdx,qword ptr [rsp+0000000000000088h]
                    00 00
  0000000000000090: 4F 8D 3C 18        lea         r15,[r8+r11]
  0000000000000094: 48 8B 8C 24 80 00  mov         rcx,qword ptr [rsp+0000000000000080h]
                    00 00
  000000000000009C: 45 33 ED           xor         r13d,r13d
  000000000000009F: 8B 84 24 A0 00 00  mov         eax,dword ptr [rsp+00000000000000A0h]
                    00
  00000000000000A6: 45 33 C9           xor         r9d,r9d
  00000000000000A9: 4C 89 4C 24 08     mov         qword ptr [rsp+8],r9
  00000000000000AE: 49 8B D8           mov         rbx,r8
  00000000000000B1: 48 8B EA           mov         rbp,rdx
  00000000000000B4: 48 89 04 24        mov         qword ptr [rsp],rax
  00000000000000B8: 48 8B F9           mov         rdi,rcx
  00000000000000BB: 0F 1F 44 00 00     nop         dword ptr [rax+rax]
  00000000000000C0: 48 85 D2           test        rdx,rdx
  00000000000000C3: 74 07              je          00000000000000CC
  00000000000000C5: F3 0F 10 45 00     movss       xmm0,dword ptr [rbp]
  00000000000000CA: EB 03              jmp         00000000000000CF
  00000000000000CC: 0F 57 C0           xorps       xmm0,xmm0
  00000000000000CF: 45 33 C0           xor         r8d,r8d
  00000000000000D2: F3 41 0F 11 07     movss       dword ptr [r15],xmm0
  00000000000000D7: 45 85 D2           test        r10d,r10d
  00000000000000DA: 0F 8E CD 00 00 00  jle         00000000000001AD
  00000000000000E0: 41 83 FA 08        cmp         r10d,8
  00000000000000E4: 0F 82 C3 00 00 00  jb          00000000000001AD
  00000000000000EA: F3 42 0F 10 24 1B  movss       xmm4,dword ptr [rbx+r11]
  00000000000000F0: 45 8D 4A FF        lea         r9d,[r10-1]
  00000000000000F4: 43 8D 04 29        lea         eax,[r9+r13]
  00000000000000F8: 48 63 C8           movsxd      rcx,eax
  00000000000000FB: 49 8D 52 FF        lea         rdx,[r10-1]
  00000000000000FF: 48 8B 84 24 80 00  mov         rax,qword ptr [rsp+0000000000000080h]
                    00 00
  0000000000000107: 49 8D 14 93        lea         rdx,[r11+rdx*4]
  000000000000010B: 48 8D 04 88        lea         rax,[rax+rcx*4]
  000000000000010F: 4C 3B D8           cmp         r11,rax
  0000000000000112: 77 09              ja          000000000000011D
  0000000000000114: 48 3B D7           cmp         rdx,rdi
  0000000000000117: 0F 83 7F 00 00 00  jae         000000000000019C
  000000000000011D: 8B 84 24 90 00 00  mov         eax,dword ptr [rsp+0000000000000090h]
                    00
  0000000000000124: 41 03 C1           add         eax,r9d
  0000000000000127: 48 63 C8           movsxd      rcx,eax
  000000000000012A: 49 8D 04 8E        lea         rax,[r14+rcx*4]
  000000000000012E: 4C 3B D8           cmp         r11,rax
  0000000000000131: 77 09              ja          000000000000013C
  0000000000000133: 4B 8D 04 A6        lea         rax,[r14+r12*4]
  0000000000000137: 48 3B D0           cmp         rdx,rax
  000000000000013A: 73 60              jae         000000000000019C
  000000000000013C: 0F 57 D2           xorps       xmm2,xmm2
  000000000000013F: 0F 57 DB           xorps       xmm3,xmm3
  0000000000000142: 49 8B D2           mov         rdx,r10
  0000000000000145: 4B 8D 0C A6        lea         rcx,[r14+r12*4]
  0000000000000149: 48 83 E2 F8        and         rdx,0FFFFFFFFFFFFFFF8h
  000000000000014D: 48 8B C7           mov         rax,rdi
  0000000000000150: 0F 10 00           movups      xmm0,xmmword ptr [rax]
  0000000000000153: 49 83 C0 08        add         r8,8
  0000000000000157: 0F 10 09           movups      xmm1,xmmword ptr [rcx]
  000000000000015A: 0F 59 C8           mulps       xmm1,xmm0
  000000000000015D: 0F 10 40 10        movups      xmm0,xmmword ptr [rax+10h]
  0000000000000161: 48 83 C0 20        add         rax,20h
  0000000000000165: 0F 58 D1           addps       xmm2,xmm1
  0000000000000168: 0F 10 49 10        movups      xmm1,xmmword ptr [rcx+10h]
  000000000000016C: 48 83 C1 20        add         rcx,20h
  0000000000000170: 0F 59 C8           mulps       xmm1,xmm0
  0000000000000173: 0F 58 D9           addps       xmm3,xmm1
  0000000000000176: 4C 3B C2           cmp         r8,rdx
  0000000000000179: 7C D5              jl          0000000000000150
  000000000000017B: 0F 58 D3           addps       xmm2,xmm3
  000000000000017E: 0F 28 CA           movaps      xmm1,xmm2
  0000000000000181: 0F 12 CA           movhlps     xmm1,xmm2
  0000000000000184: 0F 58 CA           addps       xmm1,xmm2
  0000000000000187: 0F 28 C1           movaps      xmm0,xmm1
  000000000000018A: 0F C6 C1 F5        shufps      xmm0,xmm1,0F5h
  000000000000018E: F3 0F 58 C8        addss       xmm1,xmm0
  0000000000000192: F3 0F 58 E1        addss       xmm4,xmm1
  0000000000000196: F3 42 0F 11 24 1B  movss       dword ptr [rbx+r11],xmm4
  000000000000019C: 4C 8B 4C 24 08     mov         r9,qword ptr [rsp+8]
  00000000000001A1: 48 8B 04 24        mov         rax,qword ptr [rsp]
  00000000000001A5: 48 8B 8C 24 80 00  mov         rcx,qword ptr [rsp+0000000000000080h]
                    00 00
  00000000000001AD: 4D 3B C2           cmp         r8,r10
  00000000000001B0: 7D 42              jge         00000000000001F4
  00000000000001B2: F3 42 0F 10 0C 1B  movss       xmm1,dword ptr [rbx+r11]
  00000000000001B8: 4B 8D 04 01        lea         rax,[r9+r8]
  00000000000001BC: 48 8D 0C 81        lea         rcx,[rcx+rax*4]
  00000000000001C0: 4B 8D 04 04        lea         rax,[r12+r8]
  00000000000001C4: 49 8D 14 86        lea         rdx,[r14+rax*4]
  00000000000001C8: 49 8B C2           mov         rax,r10
  00000000000001CB: 49 2B C0           sub         rax,r8
  00000000000001CE: 66 90              nop
  00000000000001D0: F3 0F 10 02        movss       xmm0,dword ptr [rdx]
  00000000000001D4: 48 83 C2 04        add         rdx,4
  00000000000001D8: F3 0F 59 01        mulss       xmm0,dword ptr [rcx]
  00000000000001DC: 48 83 C1 04        add         rcx,4
  00000000000001E0: F3 0F 58 C8        addss       xmm1,xmm0
  00000000000001E4: 48 83 E8 01        sub         rax,1
  00000000000001E8: 75 E6              jne         00000000000001D0
  00000000000001EA: 48 8B 04 24        mov         rax,qword ptr [rsp]
  00000000000001EE: F3 42 0F 11 0C 1B  movss       dword ptr [rbx+r11],xmm1
  00000000000001F4: 48 8B 94 24 88 00  mov         rdx,qword ptr [rsp+0000000000000088h]
                    00 00
  00000000000001FC: 4A 8D 0C 95 00 00  lea         rcx,[r10*4+0000000000000000h]
                    00 00
  0000000000000204: 48 03 F9           add         rdi,rcx
  0000000000000207: 4D 03 CA           add         r9,r10
  000000000000020A: 48 8B 8C 24 80 00  mov         rcx,qword ptr [rsp+0000000000000080h]
                    00 00
  0000000000000212: 49 83 C7 04        add         r15,4
  0000000000000216: 45 03 EA           add         r13d,r10d
  0000000000000219: 4C 89 4C 24 08     mov         qword ptr [rsp+8],r9
  000000000000021E: 48 83 C5 04        add         rbp,4
  0000000000000222: 48 83 C3 04        add         rbx,4
  0000000000000226: 48 83 E8 01        sub         rax,1
  000000000000022A: 48 89 04 24        mov         qword ptr [rsp],rax
  000000000000022E: 0F 85 8C FE FF FF  jne         00000000000000C0
  0000000000000234: 8B 8C 24 90 00 00  mov         ecx,dword ptr [rsp+0000000000000090h]
                    00
  000000000000023B: 4C 8B 44 24 10     mov         r8,qword ptr [rsp+10h]
  0000000000000240: 8B 94 24 A0 00 00  mov         edx,dword ptr [rsp+00000000000000A0h]
                    00
  0000000000000247: 48 8B 44 24 18     mov         rax,qword ptr [rsp+18h]
  000000000000024C: 4C 8B 4C 24 20     mov         r9,qword ptr [rsp+20h]
  0000000000000251: 41 03 CA           add         ecx,r10d
  0000000000000254: 4D 03 C1           add         r8,r9
  0000000000000257: 4D 03 E2           add         r12,r10
  000000000000025A: 89 8C 24 90 00 00  mov         dword ptr [rsp+0000000000000090h],ecx
                    00
  0000000000000261: 48 83 E8 01        sub         rax,1
  0000000000000265: 4C 89 44 24 10     mov         qword ptr [rsp+10h],r8
  000000000000026A: 48 89 44 24 18     mov         qword ptr [rsp+18h],rax
  000000000000026F: 0F 85 0B FE FF FF  jne         0000000000000080
  0000000000000275: 4C 8B 7C 24 30     mov         r15,qword ptr [rsp+30h]
  000000000000027A: 4C 8B 6C 24 38     mov         r13,qword ptr [rsp+38h]
  000000000000027F: 4C 8B 64 24 40     mov         r12,qword ptr [rsp+40h]
  0000000000000284: 48 8B 7C 24 48     mov         rdi,qword ptr [rsp+48h]
  0000000000000289: 48 8B 74 24 50     mov         rsi,qword ptr [rsp+50h]
  000000000000028E: 48 8B 6C 24 58     mov         rbp,qword ptr [rsp+58h]
  0000000000000293: 48 8B 5C 24 70     mov         rbx,qword ptr [rsp+70h]
  0000000000000298: 48 83 C4 60        add         rsp,60h
  000000000000029C: 41 5E              pop         r14
  000000000000029E: C3                 ret

  Summary

          50 .chks64
          6C .debug$S
          2F .drectve
          24 .pdata
         29F .text$mn
          44 .xdata
