using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using static kernel32;

public static partial class std {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float min(float x, float y) {
        return (float)Math.Min(x, y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong max(ulong x, ulong y) {
        return Math.Max(x, y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float sqrtf(float x) {
        return (float)Math.Sqrt(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float powf(float x, float y) {
        return (float)Math.Pow(x, y);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float logf(float x) {
        return (float)Math.Log(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float expf(float x) {
        return (float)Math.Exp(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float tanhf(float x) {
        return (float)Math.Tanh(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float coshf(float x) {
        return (float)Math.Cosh(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float fabsf(float x) {
        return (float)Math.Abs(x);
    }

    static unsafe ulong xorshift32(ulong* state) {
        /* See href="https://en.wikipedia.org/wiki/Xorshift#xorshift.2A" */
        unchecked {
            *state ^= *state >> 12;
            *state ^= *state << 25;
            *state ^= *state >> 27;
            return (*state * 0x2545F4914F6CDD1Dul) >> 32;
        }
    }

    /// <summary>
    /// (-2147483648, +2147483647)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe int rand(ulong* state) {
        unchecked {
            return (int)(xorshift32(state));
        }
    }

    /// <summary>
    /// (0, +4294967295)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe uint urand(ulong* state) {
        unchecked {
            return (uint)(xorshift32(state));
        }
    }

    /// <summary>
    /// (-1, +1)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float randf(ulong* state) {
        unchecked {
            return (int)xorshift32(state) / 2147483647.0f;
        }
    }

    /// <summary>
    /// (0, +1)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float urandf(ulong* state) {
        unchecked {
            return (uint)xorshift32(state) / 4294967295.0f;
        }
    }

    /// <summary>
    /// (-6.2831853071795862, +6.2831853071795862)
    /// </summary>
    public static unsafe float normal(ulong* state, float mean = 0, float std = 1) {
        // Box–Muller transform
        return (float)((std * Math.Sqrt(-2.0 * Math.Log(urandf(state) + 1e-12f))) * Math.Sin(2 * 3.1415926535897931 * urandf(state)) + mean);
    }


    public static void fclose(IntPtr hFile) {
        if (hFile != IntPtr.Zero && hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
        }
    }

    public static IntPtr fopen(string fileName, string mode = "rb") {
        CreationDisposition nCreationDisposition = CreationDisposition.OpenExisting;
        uint dwDesiredAccess = GENERIC_READ;
        switch (mode) {
            case "r":
            case "rb":
                break;
            case "r+":
            case "r+b":
            case "rb+":
                dwDesiredAccess = GENERIC_WRITE;
                nCreationDisposition = CreationDisposition.OpenExisting;
                break;
            case "w":
                dwDesiredAccess = GENERIC_WRITE;
                nCreationDisposition = CreationDisposition.CreateAlways;
                break;
            default:
                throw new NotSupportedException($"The specified mode '{mode}' is not supported.");
        }
        var hFile = CreateFile(Path.GetFullPath(fileName),
                     dwDesiredAccess,
                     ShareMode.Read,
                     IntPtr.Zero,
                     nCreationDisposition,
                     FILE_ATTRIBUTE_NORMAL,
                     IntPtr.Zero);
        if (hFile == INVALID_HANDLE_VALUE) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            const int ERROR_FILE_NOT_FOUND = 2;
            if (ERROR_FILE_NOT_FOUND == lastWin32Error) {
                throw new FileNotFoundException("File not found.", fileName);
            }
            throw new Win32Exception(lastWin32Error);
        }
        return hFile;
    }

    public unsafe static int fwrite(byte[] _Buffer, int count, IntPtr hFile) { fixed (void* ptr = _Buffer) { return fwrite(ptr, sizeof(byte), count, hFile); } }
    public unsafe static int fwrite(void* _Buffer, int _ElementSize, int _ElementCount, IntPtr hFile) {
        int nNumberOfBytesToWrite = _ElementSize * _ElementCount;
        if (nNumberOfBytesToWrite == 0) {
            return 0;
        }
        int bResult = WriteFile(
            hFile,
            _Buffer,
            nNumberOfBytesToWrite,
            out int numberOfBytesWritten,
            IntPtr.Zero);
        if (bResult == 0) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            if (lastWin32Error == 232) {
                return 0;
            }
            throw new Win32Exception(lastWin32Error);
        }
        return numberOfBytesWritten;
    }

    public unsafe static uint fread(void* _Buffer, uint _ElementSize, uint _ElementCount, IntPtr hFile) {
        uint nNumberOfBytesToRead = _ElementSize * _ElementCount;
        if (nNumberOfBytesToRead == 0) {
            return 0;
        }
        const int ERROR_BROKEN_PIPE = 109;
        int bResult = ReadFile(
            hFile,
            _Buffer,
            nNumberOfBytesToRead,
            out uint numberOfBytesRead,
            IntPtr.Zero);
        if (bResult == 0) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            if (lastWin32Error == ERROR_BROKEN_PIPE) {
                return 0;
            }
            throw new Win32Exception(lastWin32Error);
        }
        return numberOfBytesRead;
    }

    public unsafe static ulong fseek(IntPtr hFile, long offset, SeekOrigin origin) {
        int lastWin32Error;
        int lo = (int)offset,
           hi = (int)(offset >> 32);
        lo = SetFilePointerWin32(hFile, lo, &hi, (int)origin);
        if (lo == -1 && (lastWin32Error = Marshal.GetLastWin32Error()) != 0) {
            throw new Win32Exception(lastWin32Error);
        }
        return (((ulong)(uint)hi << 32) | (uint)lo);
    }

    public unsafe static ulong ftell(IntPtr hFile) {
        int lastWin32Error;
        int hi = 0;
        int lo = SetFilePointerWin32(hFile, 0, &hi, (int)SeekOrigin.Current);
        if (lo == -1 && (lastWin32Error = Marshal.GetLastWin32Error()) != 0) {
            throw new Win32Exception(lastWin32Error);
        }
        return (((ulong)(uint)hi << 32) | (uint)lo);
    }

    public static ulong fsize(IntPtr hFile) {
        int lowSize = GetFileSize(hFile, out int highSize);
        if (lowSize == -1) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            throw new Win32Exception(lastWin32Error);
        }
        return ((ulong)highSize << 32) | (uint)lowSize;
    }

    public static unsafe void* malloc(uint _ElementCount, uint _ElementSize) {
        return malloc((ulong)_ElementCount * _ElementSize);
    }

    public static unsafe void* malloc(ulong size) {
        if (size <= 0) throw new ArgumentOutOfRangeException("size");
        var hglobal = LocalAlloc(LMEM_FIXED, new UIntPtr(size));
        if (hglobal == null) {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        return hglobal;
    }

    public static unsafe void* realloc(void* hglobal, ulong size) {
        var hglobal_ = LocalReAlloc(hglobal, new UIntPtr(size), LMEM_MOVEABLE);
        if (hglobal_ == null) {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        return hglobal_;
    }

    public static unsafe void free(void* hglobal) {
        if (hglobal != null)
            LocalFree(hglobal);
    }

    public static unsafe void memcpy(void* destination, void* source, ulong size) {
        if (size < 0) throw new ArgumentOutOfRangeException("size");
        CopyMemory(
            destination,
            source,
            new UIntPtr(size));
    }

    public static unsafe void memset(void* destination, byte fill, ulong size) {
        if (size < 0) throw new ArgumentOutOfRangeException("size");
        FillMemory(
            destination,
            new UIntPtr(size),
            fill);
    }

    public static string memsize(ulong size) {
        string[] sizes = { "B", "KiB", "MiB", "GiB", "TiB" };
        double len = size;
        int order = 0;
        while (len >= 1024 && order < sizes.Length - 1) {
            order++;
            len = len / 1024;
        }
        return string.Format("{0:0.##} {1}", len, sizes[order]);
    }

    public static void printf(string fmt, params object[] args) {
        fprintf(Console.Out, fmt, args);
    }

    public static void fprintf(TextWriter _Stream, string fmt, params object[] args) {
        for (int i = 0; i < args.Length; i++) {
            var pos = fmt.IndexOf("%");
            if (pos < 0 || pos + 1 >= fmt.Length) {
                throw new ArgumentOutOfRangeException();
            }
            string s = fmt.Substring(
                0,
                pos);
            int skip = 2;
            switch (fmt[pos + 1]) {
                case 'f':
                    if (pos + 2 < fmt.Length && char.IsDigit(fmt[pos + 2])) {
                        s += "{" + i.ToString() + ":F" + fmt[pos + 2] + "}";
                        skip++;
                    } else {
                        s += "{" + i.ToString() + ":F6}";
                    }
                    break;
                case 'x':
                    if (pos + 2 < fmt.Length && char.IsDigit(fmt[pos + 2])) {
                        s += "{" + i.ToString() + ":x" + fmt[pos + 2] + "}";
                        skip++;
                    } else {
                        s += "{" + i.ToString() + ":x}";
                    }
                    break;
                case 'z':
                    s += "{" + i.ToString() + "}";
                    if (pos + 2 < fmt.Length && fmt[pos + 2] == 'u') {
                        skip++;
                    }
                    break;
                case 'l':
                    s += "{" + i.ToString() + "}";
                    if (pos + 2 < fmt.Length && fmt[pos + 2] == 'l') {
                        skip++;
                    }
                    if (pos + 3 < fmt.Length && (fmt[pos + 3] == 'd' || fmt[pos + 3] == 'u')) {
                        skip++;
                    }
                    break;
                case 'd':
                case 's':
                case 'g':
                case 'e':
                    s += "{" + i.ToString() + "}";
                    break;
                default:
                    throw new NotImplementedException();
            }
            s += fmt.Substring(
                pos + skip);
            fmt = s;
        }
        _Stream.Write(fmt, args);
    }
}