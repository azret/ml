Microsoft (R) COFF/PE Dumper Version 14.39.33523.0
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file MatMul.obj

File Type: COFF OBJECT

matmul_backward:
  0000000000000000: 40 53              push        rbx
  0000000000000002: 56                 push        rsi
  0000000000000003: 57                 push        rdi
  0000000000000004: 41 56              push        r14
  0000000000000006: 48 83 EC 58        sub         rsp,58h
  000000000000000A: 8B 41 40           mov         eax,dword ptr [rcx+40h]
  000000000000000D: 45 33 C0           xor         r8d,r8d
  0000000000000010: 48 8B 59 08        mov         rbx,qword ptr [rcx+8]
  0000000000000014: 4C 8B 49 10        mov         r9,qword ptr [rcx+10h]
  0000000000000018: 4C 8B 71 18        mov         r14,qword ptr [rcx+18h]
  000000000000001C: 48 8B 79 20        mov         rdi,qword ptr [rcx+20h]
  0000000000000020: 48 8B 71 28        mov         rsi,qword ptr [rcx+28h]
  0000000000000024: 4C 8B 51 38        mov         r10,qword ptr [rcx+38h]
  0000000000000028: 8B 51 48           mov         edx,dword ptr [rcx+48h]
  000000000000002B: 44 8B 59 44        mov         r11d,dword ptr [rcx+44h]
  000000000000002F: 48 89 5C 24 08     mov         qword ptr [rsp+8],rbx
  0000000000000034: 4C 89 4C 24 30     mov         qword ptr [rsp+30h],r9
  0000000000000039: 4C 89 74 24 28     mov         qword ptr [rsp+28h],r14
  000000000000003E: 48 89 7C 24 10     mov         qword ptr [rsp+10h],rdi
  0000000000000043: 48 89 74 24 18     mov         qword ptr [rsp+18h],rsi
  0000000000000048: 4C 89 54 24 20     mov         qword ptr [rsp+20h],r10
  000000000000004D: 89 84 24 98 00 00  mov         dword ptr [rsp+0000000000000098h],eax
                    00
  0000000000000054: 89 94 24 88 00 00  mov         dword ptr [rsp+0000000000000088h],edx
                    00
  000000000000005B: 44 89 84 24 80 00  mov         dword ptr [rsp+0000000000000080h],r8d
                    00 00
  0000000000000063: 85 C0              test        eax,eax
  0000000000000065: 0F 84 CB 02 00 00  je          0000000000000336
  000000000000006B: 4C 89 64 24 50     mov         qword ptr [rsp+50h],r12
  0000000000000070: 4C 89 6C 24 48     mov         qword ptr [rsp+48h],r13
  0000000000000075: 4C 89 7C 24 40     mov         qword ptr [rsp+40h],r15
  000000000000007A: 66 0F 1F 44 00 00  nop         word ptr [rax+rax]
  0000000000000080: 41 8B C0           mov         eax,r8d
  0000000000000083: 45 33 ED           xor         r13d,r13d
  0000000000000086: 41 0F AF C3        imul        eax,r11d
  000000000000008A: 4D 8D 24 81        lea         r12,[r9+rax*4]
  000000000000008E: 4D 8D 0C 86        lea         r9,[r14+rax*4]
  0000000000000092: 85 D2              test        edx,edx
  0000000000000094: 0F 84 6F 02 00 00  je          0000000000000309
  000000000000009A: 41 8B C0           mov         eax,r8d
  000000000000009D: 4C 89 14 24        mov         qword ptr [rsp],r10
  00000000000000A1: 0F AF C2           imul        eax,edx
  00000000000000A4: 4D 8B FA           mov         r15,r10
  00000000000000A7: 89 84 24 90 00 00  mov         dword ptr [rsp+0000000000000090h],eax
                    00
  00000000000000AE: 66 90              nop
  00000000000000B0: 41 03 C5           add         eax,r13d
  00000000000000B3: 45 33 C0           xor         r8d,r8d
  00000000000000B6: C5 FA 10 14 83     vmovss      xmm2,dword ptr [rbx+rax*4]
  00000000000000BB: 41 8B C5           mov         eax,r13d
  00000000000000BE: 33 DB              xor         ebx,ebx
  00000000000000C0: 41 0F AF C3        imul        eax,r11d
  00000000000000C4: C5 F8 28 DA        vmovaps     xmm3,xmm2
  00000000000000C8: C4 E2 7D 18 DA     vbroadcastss ymm3,xmm2
  00000000000000CD: 4C 8D 14 87        lea         r10,[rdi+rax*4]
  00000000000000D1: 48 8D 14 86        lea         rdx,[rsi+rax*4]
  00000000000000D5: 45 85 DB           test        r11d,r11d
  00000000000000D8: 0F 84 D9 01 00 00  je          00000000000002B7
  00000000000000DE: 41 83 FB 08        cmp         r11d,8
  00000000000000E2: 0F 82 97 00 00 00  jb          000000000000017F
  00000000000000E8: 41 8D 43 FF        lea         eax,[r11-1]
  00000000000000EC: 48 63 C8           movsxd      rcx,eax
  00000000000000EF: 49 8D 34 8C        lea         rsi,[r12+rcx*4]
  00000000000000F3: 48 8D 04 8A        lea         rax,[rdx+rcx*4]
  00000000000000F7: 48 3B D6           cmp         rdx,rsi
  00000000000000FA: 77 09              ja          0000000000000105
  00000000000000FC: 49 3B C4           cmp         rax,r12
  00000000000000FF: 0F 83 7A 00 00 00  jae         000000000000017F
  0000000000000105: 49 8D 3C 89        lea         rdi,[r9+rcx*4]
  0000000000000109: 48 3B D7           cmp         rdx,rdi
  000000000000010C: 77 05              ja          0000000000000113
  000000000000010E: 49 3B C1           cmp         rax,r9
  0000000000000111: 73 6C              jae         000000000000017F
  0000000000000113: 49 8D 0C 8A        lea         rcx,[r10+rcx*4]
  0000000000000117: 48 3B D1           cmp         rdx,rcx
  000000000000011A: 77 05              ja          0000000000000121
  000000000000011C: 49 3B C2           cmp         rax,r10
  000000000000011F: 73 5E              jae         000000000000017F
  0000000000000121: 4C 3B CE           cmp         r9,rsi
  0000000000000124: 77 05              ja          000000000000012B
  0000000000000126: 49 3B FC           cmp         rdi,r12
  0000000000000129: 73 54              jae         000000000000017F
  000000000000012B: 4C 3B C9           cmp         r9,rcx
  000000000000012E: 77 05              ja          0000000000000135
  0000000000000130: 49 3B FA           cmp         rdi,r10
  0000000000000133: 73 4A              jae         000000000000017F
  0000000000000135: 41 8B FB           mov         edi,r11d
  0000000000000138: 83 E7 F8           and         edi,0FFFFFFF8h
  000000000000013B: 49 8B F2           mov         rsi,r10
  000000000000013E: 4D 8B F4           mov         r14,r12
  0000000000000141: 49 2B F1           sub         rsi,r9
  0000000000000144: 4D 2B F1           sub         r14,r9
  0000000000000147: 48 8B CA           mov         rcx,rdx
  000000000000014A: 49 8B C1           mov         rax,r9
  000000000000014D: 49 2B C9           sub         rcx,r9
  0000000000000150: C5 FC 10 08        vmovups     ymm1,ymmword ptr [rax]
  0000000000000154: C4 E2 65 B8 0C 06  vfmadd231ps ymm1,ymm3,ymmword ptr [rsi+rax]
  000000000000015A: C5 FC 11 08        vmovups     ymmword ptr [rax],ymm1
  000000000000015E: C5 FC 10 0C 01     vmovups     ymm1,ymmword ptr [rcx+rax]
  0000000000000163: C4 C2 65 B8 0C 06  vfmadd231ps ymm1,ymm3,ymmword ptr [r14+rax]
  0000000000000169: 41 83 C0 08        add         r8d,8
  000000000000016D: 48 83 C3 08        add         rbx,8
  0000000000000171: C5 FC 11 0C 01     vmovups     ymmword ptr [rcx+rax],ymm1
  0000000000000176: 48 8D 40 20        lea         rax,[rax+20h]
  000000000000017A: 44 3B C7           cmp         r8d,edi
  000000000000017D: 72 D1              jb          0000000000000150
  000000000000017F: 45 3B C3           cmp         r8d,r11d
  0000000000000182: 0F 83 2F 01 00 00  jae         00000000000002B7
  0000000000000188: 41 8B C3           mov         eax,r11d
  000000000000018B: 41 2B C0           sub         eax,r8d
  000000000000018E: 83 F8 04           cmp         eax,4
  0000000000000191: 0F 82 D9 00 00 00  jb          0000000000000270
  0000000000000197: 41 8B C3           mov         eax,r11d
  000000000000019A: 48 8D 4B 01        lea         rcx,[rbx+1]
  000000000000019E: 41 2B C0           sub         eax,r8d
  00000000000001A1: 49 8D 0C 89        lea         rcx,[r9+rcx*4]
  00000000000001A5: 83 E8 04           sub         eax,4
  00000000000001A8: 48 8B FA           mov         rdi,rdx
  00000000000001AB: C1 E8 02           shr         eax,2
  00000000000001AE: 49 8B F2           mov         rsi,r10
  00000000000001B1: 4D 8B F4           mov         r14,r12
  00000000000001B4: 49 2B F9           sub         rdi,r9
  00000000000001B7: 49 2B F1           sub         rsi,r9
  00000000000001BA: 4D 2B F1           sub         r14,r9
  00000000000001BD: FF C0              inc         eax
  00000000000001BF: 44 8B F8           mov         r15d,eax
  00000000000001C2: 45 8D 04 80        lea         r8d,[r8+rax*4]
  00000000000001C6: 48 8D 1C 83        lea         rbx,[rbx+rax*4]
  00000000000001CA: 66 0F 1F 44 00 00  nop         word ptr [rax+rax]
  00000000000001D0: C5 FA 10 44 31 FC  vmovss      xmm0,dword ptr [rcx+rsi-4]
  00000000000001D6: C4 E2 69 A9 41 FC  vfmadd213ss xmm0,xmm2,dword ptr [rcx-4]
  00000000000001DC: C5 FA 11 41 FC     vmovss      dword ptr [rcx-4],xmm0
  00000000000001E1: C4 C1 7A 10 44 0E  vmovss      xmm0,dword ptr [r14+rcx-4]
                    FC
  00000000000001E8: C4 E2 69 A9 44 0F  vfmadd213ss xmm0,xmm2,dword ptr [rdi+rcx-4]
                    FC
  00000000000001EF: C5 FA 11 44 0F FC  vmovss      dword ptr [rdi+rcx-4],xmm0
  00000000000001F5: C5 FA 10 0C 31     vmovss      xmm1,dword ptr [rcx+rsi]
  00000000000001FA: C4 E2 69 A9 09     vfmadd213ss xmm1,xmm2,dword ptr [rcx]
  00000000000001FF: C5 FA 11 09        vmovss      dword ptr [rcx],xmm1
  0000000000000203: C4 C1 7A 10 04 0E  vmovss      xmm0,dword ptr [r14+rcx]
  0000000000000209: C4 E2 69 A9 04 0F  vfmadd213ss xmm0,xmm2,dword ptr [rdi+rcx]
  000000000000020F: C5 FA 11 04 0F     vmovss      dword ptr [rdi+rcx],xmm0
  0000000000000214: C5 FA 10 4C 31 04  vmovss      xmm1,dword ptr [rcx+rsi+4]
  000000000000021A: C4 E2 69 A9 49 04  vfmadd213ss xmm1,xmm2,dword ptr [rcx+4]
  0000000000000220: C5 FA 11 49 04     vmovss      dword ptr [rcx+4],xmm1
  0000000000000225: C4 C1 7A 10 44 0E  vmovss      xmm0,dword ptr [r14+rcx+4]
                    04
  000000000000022C: C4 E2 69 A9 44 0F  vfmadd213ss xmm0,xmm2,dword ptr [rdi+rcx+4]
                    04
  0000000000000233: C5 FA 11 44 0F 04  vmovss      dword ptr [rdi+rcx+4],xmm0
  0000000000000239: C5 FA 10 4C 31 08  vmovss      xmm1,dword ptr [rcx+rsi+8]
  000000000000023F: C4 E2 69 A9 49 08  vfmadd213ss xmm1,xmm2,dword ptr [rcx+8]
  0000000000000245: C5 FA 11 49 08     vmovss      dword ptr [rcx+8],xmm1
  000000000000024A: C4 C1 7A 10 44 0E  vmovss      xmm0,dword ptr [r14+rcx+8]
                    08
  0000000000000251: C4 E2 69 A9 44 0F  vfmadd213ss xmm0,xmm2,dword ptr [rdi+rcx+8]
                    08
  0000000000000258: C5 FA 11 44 0F 08  vmovss      dword ptr [rdi+rcx+8],xmm0
  000000000000025E: 48 8D 49 10        lea         rcx,[rcx+10h]
  0000000000000262: 49 83 EF 01        sub         r15,1
  0000000000000266: 0F 85 64 FF FF FF  jne         00000000000001D0
  000000000000026C: 4C 8B 3C 24        mov         r15,qword ptr [rsp]
  0000000000000270: 45 3B C3           cmp         r8d,r11d
  0000000000000273: 73 42              jae         00000000000002B7
  0000000000000275: 49 8D 0C 99        lea         rcx,[r9+rbx*4]
  0000000000000279: 4D 2B D1           sub         r10,r9
  000000000000027C: 49 8B DC           mov         rbx,r12
  000000000000027F: 49 2B D1           sub         rdx,r9
  0000000000000282: 49 2B D9           sub         rbx,r9
  0000000000000285: 41 8B C3           mov         eax,r11d
  0000000000000288: 41 2B C0           sub         eax,r8d
  000000000000028B: 44 8B C0           mov         r8d,eax
  000000000000028E: C4 A1 7A 10 04 11  vmovss      xmm0,dword ptr [rcx+r10]
  0000000000000294: C4 E2 69 A9 01     vfmadd213ss xmm0,xmm2,dword ptr [rcx]
  0000000000000299: C5 FA 11 01        vmovss      dword ptr [rcx],xmm0
  000000000000029D: C5 FA 10 0C 0B     vmovss      xmm1,dword ptr [rbx+rcx]
  00000000000002A2: C4 E2 69 A9 0C 0A  vfmadd213ss xmm1,xmm2,dword ptr [rdx+rcx]
  00000000000002A8: C5 FA 11 0C 0A     vmovss      dword ptr [rdx+rcx],xmm1
  00000000000002AD: 48 8D 49 04        lea         rcx,[rcx+4]
  00000000000002B1: 49 83 E8 01        sub         r8,1
  00000000000002B5: 75 D7              jne         000000000000028E
  00000000000002B7: 4C 8B 54 24 20     mov         r10,qword ptr [rsp+20h]
  00000000000002BC: 4D 85 D2           test        r10,r10
  00000000000002BF: 74 0A              je          00000000000002CB
  00000000000002C1: C4 C1 6A 58 07     vaddss      xmm0,xmm2,dword ptr [r15]
  00000000000002C6: C4 C1 7A 11 07     vmovss      dword ptr [r15],xmm0
  00000000000002CB: 8B 94 24 88 00 00  mov         edx,dword ptr [rsp+0000000000000088h]
                    00
  00000000000002D2: 49 83 C7 04        add         r15,4
  00000000000002D6: 8B 84 24 90 00 00  mov         eax,dword ptr [rsp+0000000000000090h]
                    00
  00000000000002DD: 41 FF C5           inc         r13d
  00000000000002E0: 48 8B 5C 24 08     mov         rbx,qword ptr [rsp+8]
  00000000000002E5: 48 8B 7C 24 10     mov         rdi,qword ptr [rsp+10h]
  00000000000002EA: 48 8B 74 24 18     mov         rsi,qword ptr [rsp+18h]
  00000000000002EF: 4C 89 3C 24        mov         qword ptr [rsp],r15
  00000000000002F3: 44 3B EA           cmp         r13d,edx
  00000000000002F6: 0F 82 B4 FD FF FF  jb          00000000000000B0
  00000000000002FC: 44 8B 84 24 80 00  mov         r8d,dword ptr [rsp+0000000000000080h]
                    00 00
  0000000000000304: 4C 8B 74 24 28     mov         r14,qword ptr [rsp+28h]
  0000000000000309: 4C 8B 4C 24 30     mov         r9,qword ptr [rsp+30h]
  000000000000030E: 41 FF C0           inc         r8d
  0000000000000311: 44 89 84 24 80 00  mov         dword ptr [rsp+0000000000000080h],r8d
                    00 00
  0000000000000319: 44 3B 84 24 98 00  cmp         r8d,dword ptr [rsp+0000000000000098h]
                    00 00
  0000000000000321: 0F 82 59 FD FF FF  jb          0000000000000080
  0000000000000327: 4C 8B 7C 24 40     mov         r15,qword ptr [rsp+40h]
  000000000000032C: 4C 8B 6C 24 48     mov         r13,qword ptr [rsp+48h]
  0000000000000331: 4C 8B 64 24 50     mov         r12,qword ptr [rsp+50h]
  0000000000000336: C5 F8 77           vzeroupper
  0000000000000339: 48 83 C4 58        add         rsp,58h
  000000000000033D: 41 5E              pop         r14
  000000000000033F: 5F                 pop         rdi
  0000000000000340: 5E                 pop         rsi
  0000000000000341: 5B                 pop         rbx
  0000000000000342: C3                 ret

matmul_forward_kernel:
  0000000000000000: 40 57              push        rdi
  0000000000000002: 44 8B 59 48        mov         r11d,dword ptr [rcx+48h]
  0000000000000006: 8B C2              mov         eax,edx
  0000000000000008: 48 8B 79 20        mov         rdi,qword ptr [rcx+20h]
  000000000000000C: 33 D2              xor         edx,edx
  000000000000000E: 44 8B 49 44        mov         r9d,dword ptr [rcx+44h]
  0000000000000012: 4C 8B D1           mov         r10,rcx
  0000000000000015: 41 F7 F3           div         eax,r11d
  0000000000000018: 3B 41 40           cmp         eax,dword ptr [rcx+40h]
  000000000000001B: 0F 83 7B 01 00 00  jae         000000000000019C
  0000000000000021: 41 3B D3           cmp         edx,r11d
  0000000000000024: 0F 83 72 01 00 00  jae         000000000000019C
  000000000000002A: 48 8B 49 10        mov         rcx,qword ptr [rcx+10h]
  000000000000002E: 44 8B C0           mov         r8d,eax
  0000000000000031: 48 89 5C 24 10     mov         qword ptr [rsp+10h],rbx
  0000000000000036: 41 0F AF C3        imul        eax,r11d
  000000000000003A: 48 89 74 24 18     mov         qword ptr [rsp+18h],rsi
  000000000000003F: 45 0F AF C1        imul        r8d,r9d
  0000000000000043: 4C 89 74 24 20     mov         qword ptr [rsp+20h],r14
  0000000000000048: 4C 89 7C 24 28     mov         qword ptr [rsp+28h],r15
  000000000000004D: 4D 8B 3A           mov         r15,qword ptr [r10]
  0000000000000050: 44 8B F0           mov         r14d,eax
  0000000000000053: 49 8B 42 30        mov         rax,qword ptr [r10+30h]
  0000000000000057: 8B F2              mov         esi,edx
  0000000000000059: 4A 8D 1C 81        lea         rbx,[rcx+r8*4]
  000000000000005D: 48 85 C0           test        rax,rax
  0000000000000060: 74 07              je          0000000000000069
  0000000000000062: C5 FA 10 1C 90     vmovss      xmm3,dword ptr [rax+rdx*4]
  0000000000000067: EB 04              jmp         000000000000006D
  0000000000000069: C5 E0 57 DB        vxorps      xmm3,xmm3,xmm3
  000000000000006D: 41 0F AF D1        imul        edx,r9d
  0000000000000071: 4C 8D 14 97        lea         r10,[rdi+rdx*4]
  0000000000000075: 33 D2              xor         edx,edx
  0000000000000077: 45 85 C9           test        r9d,r9d
  000000000000007A: 0F 84 FE 00 00 00  je          000000000000017E
  0000000000000080: 41 83 F9 10        cmp         r9d,10h
  0000000000000084: 72 56              jb          00000000000000DC
  0000000000000086: 45 8B C1           mov         r8d,r9d
  0000000000000089: 48 8D 43 20        lea         rax,[rbx+20h]
  000000000000008D: 41 83 E0 F0        and         r8d,0FFFFFFF0h
  0000000000000091: 49 8B CA           mov         rcx,r10
  0000000000000094: 48 2B CB           sub         rcx,rbx
  0000000000000097: C5 E8 57 D2        vxorps      xmm2,xmm2,xmm2
  000000000000009B: C5 D8 57 E4        vxorps      xmm4,xmm4,xmm4
  000000000000009F: 90                 nop
  00000000000000A0: C5 FC 10 48 E0     vmovups     ymm1,ymmword ptr [rax-20h]
  00000000000000A5: C4 E2 75 B8 54 08  vfmadd231ps ymm2,ymm1,ymmword ptr [rax+rcx-20h]
                    E0
  00000000000000AC: C5 FC 10 08        vmovups     ymm1,ymmword ptr [rax]
  00000000000000B0: C4 E2 75 B8 24 08  vfmadd231ps ymm4,ymm1,ymmword ptr [rax+rcx]
  00000000000000B6: 83 C2 10           add         edx,10h
  00000000000000B9: 48 8D 40 40        lea         rax,[rax+40h]
  00000000000000BD: 41 3B D0           cmp         edx,r8d
  00000000000000C0: 72 DE              jb          00000000000000A0
  00000000000000C2: C5 DC 58 C2        vaddps      ymm0,ymm4,ymm2
  00000000000000C6: C5 FF 7C C8        vhaddps     ymm1,ymm0,ymm0
  00000000000000CA: C5 F7 7C D1        vhaddps     ymm2,ymm1,ymm1
  00000000000000CE: C4 E3 7D 19 D0 01  vextractf128 xmm0,ymm2,1
  00000000000000D4: C5 F8 58 C2        vaddps      xmm0,xmm0,xmm2
  00000000000000D8: C5 E2 58 D8        vaddss      xmm3,xmm3,xmm0
  00000000000000DC: 41 3B D1           cmp         edx,r9d
  00000000000000DF: 0F 83 99 00 00 00  jae         000000000000017E
  00000000000000E5: 41 8B C1           mov         eax,r9d
  00000000000000E8: 48 63 FA           movsxd      rdi,edx
  00000000000000EB: 2B C2              sub         eax,edx
  00000000000000ED: 83 F8 04           cmp         eax,4
  00000000000000F0: 72 6B              jb          000000000000015D
  00000000000000F2: 41 8B C1           mov         eax,r9d
  00000000000000F5: 48 8D 4B 04        lea         rcx,[rbx+4]
  00000000000000F9: 2B C2              sub         eax,edx
  00000000000000FB: 48 8D 0C B9        lea         rcx,[rcx+rdi*4]
  00000000000000FF: 83 E8 04           sub         eax,4
  0000000000000102: 4D 8B C2           mov         r8,r10
  0000000000000105: C1 E8 02           shr         eax,2
  0000000000000108: 4C 2B C3           sub         r8,rbx
  000000000000010B: FF C0              inc         eax
  000000000000010D: 44 8B D8           mov         r11d,eax
  0000000000000110: 8D 14 82           lea         edx,[rdx+rax*4]
  0000000000000113: 48 8D 3C 87        lea         rdi,[rdi+rax*4]
  0000000000000117: 66 0F 1F 84 00 00  nop         word ptr [rax+rax+0000000000000000h]
                    00 00 00
  0000000000000120: C5 FA 10 41 FC     vmovss      xmm0,dword ptr [rcx-4]
  0000000000000125: C4 A2 79 B9 5C 01  vfmadd231ss xmm3,xmm0,dword ptr [rcx+r8-4]
                    FC
  000000000000012C: C5 FA 10 01        vmovss      xmm0,dword ptr [rcx]
  0000000000000130: C4 A2 79 B9 1C 01  vfmadd231ss xmm3,xmm0,dword ptr [rcx+r8]
  0000000000000136: C5 FA 10 49 04     vmovss      xmm1,dword ptr [rcx+4]
  000000000000013B: C4 A2 71 B9 5C 01  vfmadd231ss xmm3,xmm1,dword ptr [rcx+r8+4]
                    04
  0000000000000142: C5 FA 10 41 08     vmovss      xmm0,dword ptr [rcx+8]
  0000000000000147: C4 A2 79 B9 5C 01  vfmadd231ss xmm3,xmm0,dword ptr [rcx+r8+8]
                    08
  000000000000014E: 48 8D 49 10        lea         rcx,[rcx+10h]
  0000000000000152: 49 83 EB 01        sub         r11,1
  0000000000000156: 75 C8              jne         0000000000000120
  0000000000000158: 41 3B D1           cmp         edx,r9d
  000000000000015B: 73 21              jae         000000000000017E
  000000000000015D: 4C 2B D3           sub         r10,rbx
  0000000000000160: 48 8D 04 BB        lea         rax,[rbx+rdi*4]
  0000000000000164: 44 2B CA           sub         r9d,edx
  0000000000000167: 41 8B C9           mov         ecx,r9d
  000000000000016A: C5 FA 10 00        vmovss      xmm0,dword ptr [rax]
  000000000000016E: C4 C2 79 B9 1C 02  vfmadd231ss xmm3,xmm0,dword ptr [r10+rax]
  0000000000000174: 48 8D 40 04        lea         rax,[rax+4]
  0000000000000178: 48 83 E9 01        sub         rcx,1
  000000000000017C: 75 EC              jne         000000000000016A
  000000000000017E: 48 8B 5C 24 10     mov         rbx,qword ptr [rsp+10h]
  0000000000000183: 49 8D 04 36        lea         rax,[r14+rsi]
  0000000000000187: 4C 8B 74 24 20     mov         r14,qword ptr [rsp+20h]
  000000000000018C: 48 8B 74 24 18     mov         rsi,qword ptr [rsp+18h]
  0000000000000191: C4 C1 7A 11 1C 87  vmovss      dword ptr [r15+rax*4],xmm3
  0000000000000197: 4C 8B 7C 24 28     mov         r15,qword ptr [rsp+28h]
  000000000000019C: C5 F8 77           vzeroupper
  000000000000019F: 5F                 pop         rdi
  00000000000001A0: C3                 ret

  Summary

          88 .chks64
          70 .debug$S
          2F .drectve
          48 .pdata
         4E4 .text$mn
          74 .xdata
