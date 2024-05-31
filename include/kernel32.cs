using System;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;

public static class kernel32 {
    static kernel32() {
        if (IntPtr.Size != 8) {
            throw new PlatformNotSupportedException();
        }
    }

    [Flags]
    public enum AllocationTypes : uint {
        Commit = 0x1000,
        Reserve = 0x2000,
        Reset = 0x80000,
        LargePages = 0x20000000,
        Physical = 0x400000,
        TopDown = 0x100000,
        WriteWatch = 0x200000
    }

    [Flags]
    public enum MemoryProtections : uint {
        Execute = 0x10,
        ExecuteRead = 0x20,
        ExecuteReadWrite = 0x40,
        ExecuteWriteCopy = 0x80,
        NoAccess = 0x01,
        ReadOnly = 0x02,
        ReadWrite = 0x04,
        WriteCopy = 0x08,
        GuartModifierflag = 0x100,
        NoCacheModifierflag = 0x200,
        WriteCombineModifierflag = 0x400
    }

    [Flags]
    public enum FreeTypes : uint {
        Decommit = 0x4000,
        Release = 0x8000
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr VirtualAlloc(
            IntPtr lpAddress,
            IntPtr dwSize,
            AllocationTypes flAllocationType,
            MemoryProtections flProtect);

    [DllImport("kernel32")]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool VirtualFree(
        IntPtr lpAddress,
        IntPtr dwSize,
        FreeTypes flFreeType);

    [DllImport("kernel32.dll", EntryPoint = "GetTickCount64", SetLastError = false)]
    public static extern ulong millis();

    [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
    public static extern unsafe void CopyMemory(void* destination, void* source, UIntPtr length);

    [DllImport("kernel32.dll", EntryPoint = "RtlFillMemory", SetLastError = false)]
    public static extern unsafe void FillMemory(void* destination, UIntPtr length, byte fill);

    public const int LMEM_FIXED          = 0x0000;
    public const int LMEM_ZEROINIT       = 0x0040;
    public const int LPTR                = 0x0040;
    public const int LMEM_MOVEABLE       = 0x0002;

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern unsafe void* LocalAlloc(int uFlags, UIntPtr length);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern unsafe void* LocalReAlloc(void* handle, UIntPtr length, int uFlags);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern unsafe void LocalFree(void* handle);

    [DllImport("kernel32.dll", EntryPoint = "RtlZeroMemory")]
    internal static extern unsafe void ZeroMemory(void* address, UIntPtr length);

    public static IntPtr INVALID_HANDLE_VALUE = new IntPtr(-1);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern int GetFileSize(IntPtr hFile, out int dwHighSize);

    [DllImport("kernel32.dll", EntryPoint = "SetFilePointer", SetLastError = true)]
    public unsafe static extern int SetFilePointerWin32(IntPtr hFile, int lo, int* hi, int origin);

    public const uint GENERIC_READ = 0x80000000;
    public const uint GENERIC_WRITE = 0x40000000;
    public const uint GENERIC_EXECUTE = 0x20000000;
    public const uint GENERIC_ALL = 0x10000000;

    public const uint FILE_ATTRIBUTE_READONLY = 0x00000001;
    public const uint FILE_ATTRIBUTE_HIDDEN = 0x00000002;
    public const uint FILE_ATTRIBUTE_SYSTEM = 0x00000004;
    public const uint FILE_ATTRIBUTE_DIRECTORY = 0x00000010;
    public const uint FILE_ATTRIBUTE_ARCHIVE = 0x00000020;
    public const uint FILE_ATTRIBUTE_DEVICE = 0x00000040;
    public const uint FILE_ATTRIBUTE_NORMAL = 0x00000080;
    public const uint FILE_ATTRIBUTE_TEMPORARY = 0x00000100;

    [DllImport("kernel32.dll", SetLastError = true)]
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
    public static extern bool CloseHandle(IntPtr handle);

    [Flags]
    public enum ShareMode : uint {
        None = 0x00000000,
        Read = 0x00000001,
        Write = 0x00000002,
        Delete = 0x00000004
    }

    public enum CreationDisposition : uint {
        New = 1,
        CreateAlways = 2,
        OpenExisting = 3,
        OpenAlways = 4,
        TruncateExisting = 5
    }

    [DllImport("kernel32.dll", BestFitMapping = false, CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr CreateFile(
        string lpFileName,
        uint dwDesiredAccess,
        ShareMode dwShareMode,
        IntPtr lpSecurityAttributes,
        CreationDisposition dwCreationDisposition,
        uint dwFlagsAndAttributes,
        IntPtr hTemplateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    public unsafe static extern int ReadFile(
        IntPtr hFile,
        void* lpBuffer,
        uint nNumberOfBytesToRead,
        out uint lpNumberOfBytesRead,
        IntPtr lpOverlapped);

    [DllImport("kernel32.dll", SetLastError = true)]
    internal unsafe static extern int WriteFile(
        IntPtr hFile,
        void* lpBuffer,
        int nNumberOfBytesToWrite,
        out int lpNumberOfBytesWritten,
        IntPtr mustBeZero);
}