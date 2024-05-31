using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

using static std;

using BYTE = System.Byte;
using DWORD = System.UInt32;
using WORD = System.UInt16;

// https://learn.microsoft.com/en-us/windows/win32/debug/pe-format

// Use this tool to examine and extract raw byte code of kernels compiled with other compilers.

public static class coff {
    public static void Main(string[] args) {
        if (args == null || args.Length == 0) {
            printf("usage: coff [file]");
            return;
        }

        parseObj(args[0], isFull: false);

        Console.Out.Flush();

        if (Debugger.IsAttached) {
            printf("\n\nPress any key to continue...");
            Console.ReadKey();
        }
    }

    public enum MachineType : WORD {
        IMAGE_FILE_MACHINE_UNKNOWN = 0x0,
        IMAGE_FILE_MACHINE_AM33 = 0x1D3,
        IMAGE_FILE_MACHINE_AMD64 = 0x8664,
        IMAGE_FILE_MACHINE_ARM = 0x1C0,
        IMAGE_FILE_MACHINE_EBC = 0xEBC,
        IMAGE_FILE_MACHINE_I386 = 0x14C,
        IMAGE_FILE_MACHINE_IA64 = 0x200,
        IMAGE_FILE_MACHINE_M32R = 0x9041,
        IMAGE_FILE_MACHINE_MIPS16 = 0x266,
        IMAGE_FILE_MACHINE_MIPSFPU = 0x366,
        IMAGE_FILE_MACHINE_MIPSFPU16 = 0x466,
        IMAGE_FILE_MACHINE_POWERPC = 0x1F0,
        IMAGE_FILE_MACHINE_POWERPCFP = 0x1F1,
        IMAGE_FILE_MACHINE_R4000 = 0x166,
        IMAGE_FILE_MACHINE_SH3 = 0x1A2,
        IMAGE_FILE_MACHINE_SH3DSP = 0x1A3,
        IMAGE_FILE_MACHINE_SH4 = 0x1A6,
        IMAGE_FILE_MACHINE_SH5 = 0x1A8,
        IMAGE_FILE_MACHINE_THUMB = 0x1C2,
        IMAGE_FILE_MACHINE_WCEMIPSV2 = 0x169
    }

    public enum Characteristic : WORD {
        IMAGE_FILE_RELOCS_STRIPPED = 0x0001,
        IMAGE_FILE_EXECUTABLE_IMAGE = 0x0002,
        IMAGE_FILE_LINE_NUMS_STRIPPED = 0x0004,
        IMAGE_FILE_LOCAL_SYMS_STRIPPED = 0x0008,
        IMAGE_FILE_AGGRESSIVE_WS_TRIM = 0x0010,
        IMAGE_FILE_LARGE_ADDRESS_AWARE = 0x0020,
        IMAGE_FILE_BYTES_REVERSED_LO = 0x0080,
        IMAGE_FILE_32BIT_MACHINE = 0x0100,
        IMAGE_FILE_DEBUG_STRIPPED = 0x0200,
        IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP = 0x0400,
        IMAGE_FILE_NET_RUN_FROM_SWAP = 0x0800,
        IMAGE_FILE_SYSTEM = 0x1000,
        IMAGE_FILE_DLL = 0x2000,
        IMAGE_FILE_UP_SYSTEM_ONLY = 0x4000,
        IMAGE_FILE_BYTES_REVERSED_HI = 0x8000
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CoffHeader {
        public MachineType Machine;

        public ushort NumberOfSection;

        public uint TimeDateStamp;

        public uint PointerToSymbolTable;

        public uint NumberOfSymbols;

        public ushort SizeOfOptionalHeader;

        public Characteristic Characteristics;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    struct _IMAGE_FILE_HEADER {
        public MachineType Machine;
        public WORD NumberOfSections;
        public DWORD TimeDateStamp;
        public DWORD PointerToSymbolTable;
        public DWORD NumberOfSymbols;
        public WORD SizeOfOptionalHeader;
        public Characteristic Characteristics;
    }

    [Flags]
    public enum DataSectionFlags : DWORD {
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        TypeReg = 0x00000000,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        TypeDsect = 0x00000001,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        TypeNoLoad = 0x00000002,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        TypeGroup = 0x00000004,
        /// <summary>
        /// The section should not be padded to the next boundary. This flag is obsolete and is replaced by IMAGE_SCN_ALIGN_1BYTES. This is valid only for object files.
        /// </summary>
        TypeNoPadded = 0x00000008,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        TypeCopy = 0x00000010,
        /// <summary>
        /// The section contains executable code.
        /// </summary>
        ContentCode = 0x00000020,
        /// <summary>
        /// The section contains initialized data.
        /// </summary>
        ContentInitializedData = 0x00000040,
        /// <summary>
        /// The section contains uninitialized data.
        /// </summary>
        ContentUninitializedData = 0x00000080,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        LinkOther = 0x00000100,
        /// <summary>
        /// The section contains comments or other information. The .drectve section has this type. This is valid for object files only.
        /// </summary>
        LinkInfo = 0x00000200,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        TypeOver = 0x00000400,
        /// <summary>
        /// The section will not become part of the image. This is valid only for object files.
        /// </summary>
        LinkRemove = 0x00000800,
        /// <summary>
        /// The section contains COMDAT data. For more information, see section 5.5.6, COMDAT Sections (Object Only). This is valid only for object files.
        /// </summary>
        LinkComDat = 0x00001000,
        /// <summary>
        /// Reset speculative exceptions handling bits in the TLB entries for this section.
        /// </summary>
        NoDeferSpecExceptions = 0x00004000,
        /// <summary>
        /// The section contains data referenced through the global pointer (GP).
        /// </summary>
        RelativeGP = 0x00008000,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        MemPurgeable = 0x00020000,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        Memory16Bit = 0x00020000,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        MemoryLocked = 0x00040000,
        /// <summary>
        /// Reserved for future use.
        /// </summary>
        MemoryPreload = 0x00080000,
        /// <summary>
        /// Align data on a 1-byte boundary. Valid only for object files.
        /// </summary>
        Align1Bytes = 0x00100000,
        /// <summary>
        /// Align data on a 2-byte boundary. Valid only for object files.
        /// </summary>
        Align2Bytes = 0x00200000,
        /// <summary>
        /// Align data on a 4-byte boundary. Valid only for object files.
        /// </summary>
        Align4Bytes = 0x00300000,
        /// <summary>
        /// Align data on an 8-byte boundary. Valid only for object files.
        /// </summary>
        Align8Bytes = 0x00400000,
        /// <summary>
        /// Align data on a 16-byte boundary. Valid only for object files.
        /// </summary>
        Align16Bytes = 0x00500000,
        /// <summary>
        /// Align data on a 32-byte boundary. Valid only for object files.
        /// </summary>
        Align32Bytes = 0x00600000,
        /// <summary>
        /// Align data on a 64-byte boundary. Valid only for object files.
        /// </summary>
        Align64Bytes = 0x00700000,
        /// <summary>
        /// Align data on a 128-byte boundary. Valid only for object files.
        /// </summary>
        Align128Bytes = 0x00800000,
        /// <summary>
        /// Align data on a 256-byte boundary. Valid only for object files.
        /// </summary>
        Align256Bytes = 0x00900000,
        /// <summary>
        /// Align data on a 512-byte boundary. Valid only for object files.
        /// </summary>
        Align512Bytes = 0x00A00000,
        /// <summary>
        /// Align data on a 1024-byte boundary. Valid only for object files.
        /// </summary>
        Align1024Bytes = 0x00B00000,
        /// <summary>
        /// Align data on a 2048-byte boundary. Valid only for object files.
        /// </summary>
        Align2048Bytes = 0x00C00000,
        /// <summary>
        /// Align data on a 4096-byte boundary. Valid only for object files.
        /// </summary>
        Align4096Bytes = 0x00D00000,
        /// <summary>
        /// Align data on an 8192-byte boundary. Valid only for object files.
        /// </summary>
        Align8192Bytes = 0x00E00000,
        /// <summary>
        /// The section contains extended relocations.
        /// </summary>
        LinkExtendedRelocationOverflow = 0x01000000,
        /// <summary>
        /// The section can be discarded as needed.
        /// </summary>
        MemoryDiscardable = 0x02000000,
        /// <summary>
        /// The section cannot be cached.
        /// </summary>
        MemoryNotCached = 0x04000000,
        /// <summary>
        /// The section is not pageable.
        /// </summary>
        MemoryNotPaged = 0x08000000,
        /// <summary>
        /// The section can be shared in memory.
        /// </summary>
        MemoryShared = 0x10000000,
        /// <summary>
        /// The section can be executed as code.
        /// </summary>
        MemoryExecute = 0x20000000,
        /// <summary>
        /// The section can be read.
        /// </summary>
        MemoryRead = 0x40000000,
        /// <summary>
        /// The section can be written to.
        /// </summary>
        MemoryWrite = 0x80000000
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    unsafe struct _IMAGE_SECTION_HEADER {
        public const int IMAGE_SIZEOF_SHORT_NAME = 8;
        public static string GetName(_IMAGE_SECTION_HEADER* p) {
            return Encoding.UTF8.GetString(p->Name, IMAGE_SIZEOF_SHORT_NAME);
        }
        public fixed BYTE Name[IMAGE_SIZEOF_SHORT_NAME];
        public DWORD VirtualSize;
        public DWORD VirtualAddress;
        public DWORD SizeOfRawData;
        public DWORD PointerToRawData;
        public DWORD PointerToRelocations;
        public DWORD PointerToLinenumbers;
        public WORD NumberOfRelocations;
        public WORD NumberOfLinenumbers;
        public DataSectionFlags Characteristics;
        public bool IsContentCode() => (Characteristics & DataSectionFlags.ContentCode) == DataSectionFlags.ContentCode;
    }

    class Section {
        public readonly Dictionary<DWORD, string> FUNC_TABLE = new Dictionary<DWORD, string>();
        public Section(string name, _IMAGE_SECTION_HEADER header, byte[] rawBytes) {
            Name = name;
            Header = header;
            RawBytes = rawBytes;
        }
        public string Name { get; }
        public _IMAGE_SECTION_HEADER Header { get; }
        public byte[] RawBytes { get; }
    }

    //  COFF Symbol Table

    //  The symbol table in this section is inherited from the traditional COFF format.
    //  It is distinct from Microsoft Visual C++ debug information.
    //  A file can contain both a COFF symbol table and Visual C++ debug information, and the two are kept separate.
    //  Some Microsoft tools use the symbol table for limited but important purposes, such as communicating COMDAT
    //      information to the linker.
    //  Section names and file names, as well as code and data symbols, are listed in the symbol table.
    //  
    //  The location of the symbol table is indicated in the COFF header.
    //  
    //  The symbol table is an array of records, each 18 bytes long. Each record is either a standard or
    //      auxiliary symbol-table record.
    //  A standard record defines a symbol or name and has the following format.

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    unsafe struct _IMAGE_SYMBOL {
        public const int IMAGE_SIZEOF_SHORT_NAME = 8;
        // The name of the symbol, represented by a union of three structures.An array of 8 bytes is used if the
        //      name is not more than 8 bytes long. For more information, see Symbol Name Representation.
        public fixed BYTE Name[IMAGE_SIZEOF_SHORT_NAME];
        // The value that is associated with the symbol.The interpretation of this field depends
        //      on SectionNumber and StorageClass. A typical meaning is the relocatable address.
        public DWORD Value;
        // The signed integer that identifies the section, using a one-based index into the section table.
        // Some values have special meaning, as defined in section 5.4.2, "Section Number Values."
        public WORD SectionNumber;

        // https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#type-representation
        //  A number that represents type.
        //  Microsoft tools set this field to 0x20 (function) or 0x0 (not a function).
        //  For more information, see Type Representation.
        public WORD Type;

        // An enumerated value that represents storage class. For more information, see Storage Class.
        public IMAGE_SYM_CLASS StorageClass;
        // The number of auxiliary symbol table entries that follow this record.
        public BYTE NumberOfAuxSymbols;
    }

    public enum IMAGE_SYM_CLASS : BYTE {
        END_OF_FUNCTION = 0xFF,
        NULL = 0,
        AUTOMATIC = 1,
        EXTERN = 2,
        STATIC = 3,
        REGISTER = 4,
        EXTERNAL_DEF = 5,
        LABEL = 6,
        UNDEFINED_LABEL = 7,
        MEMBER_OF_STRUCT = 8,
        ARGUMENT = 9,
        STRUCT_TAG = 10,
        MEMBER_OF_UNION = 11,
        UNION_TAG = 12,
        TYPE_DEFINITION = 13,
        UNDEFINED_STATIC = 14,
        ENUM_TAG = 15,
        MEMBER_OF_ENUM = 16,
        REGISTER_PARAM = 17,
        BIT_FIELD = 18,
        BLOCK = 100,
        FUNCTION = 101,
        END_OF_STRUCT = 102,
        FILE = 103,
        SECTION = 104,
        WEAK_EXTERNAL = 105
    }

    public unsafe static void parseObj(string fileName, bool isFull) {
        var file = fopen(fileName);
        try {
            parseObj(file, isFull);
        } finally {
            fclose(file);
        }
    }

    public unsafe static void parseObj(IntPtr file, bool isFull) {
        fseek(file, 0, SeekOrigin.Begin);

        _IMAGE_FILE_HEADER _IMAGE_FILE_HEADER = new _IMAGE_FILE_HEADER();

        var cc = fread(&_IMAGE_FILE_HEADER, (uint)Marshal.SizeOf<_IMAGE_FILE_HEADER>(), 1, file);
        if (cc != Marshal.SizeOf<_IMAGE_FILE_HEADER>()) {
            throw new BadImageFormatException();
        }

        if (isFull) {
            Console.WriteLine($@"FILE HEADER VALUES
            {(DWORD)_IMAGE_FILE_HEADER.Machine:X} machine ({_IMAGE_FILE_HEADER.Machine})
               {_IMAGE_FILE_HEADER.NumberOfSections} number of sections
        {_IMAGE_FILE_HEADER.TimeDateStamp:X} time date stamp {DateTimeOffset.FromUnixTimeSeconds(_IMAGE_FILE_HEADER.TimeDateStamp).DateTime.ToLocalTime()}
             {_IMAGE_FILE_HEADER.PointerToSymbolTable:X} file pointer to symbol table
              {_IMAGE_FILE_HEADER.NumberOfSymbols} number of symbols
               {_IMAGE_FILE_HEADER.SizeOfOptionalHeader} size of optional header
               {(DWORD)_IMAGE_FILE_HEADER.Characteristics} characteristics");
        }

        Debug.Assert(_IMAGE_FILE_HEADER.SizeOfOptionalHeader == 0, "_IMAGE_FILE_HEADER.SizeOfOptionalHeader != 0");

        Dictionary<string, Section> SECTION_TABLE = new Dictionary<string, Section>();

        // Read all the section headers

        var sections = new _IMAGE_SECTION_HEADER[_IMAGE_FILE_HEADER.NumberOfSections];

        for (int sectionNo = 0; sectionNo < sections.Length; sectionNo++) {
            _IMAGE_SECTION_HEADER _IMAGE_SECTION_HEADER = sections[sectionNo];
            cc = fread(&_IMAGE_SECTION_HEADER, (uint)sizeof(_IMAGE_SECTION_HEADER), 1, file);
            if (cc != Marshal.SizeOf<_IMAGE_SECTION_HEADER>()) {
                throw new BadImageFormatException();
            }
            sections[sectionNo] = _IMAGE_SECTION_HEADER;

            if (isFull) {
                Console.WriteLine($@"
SECTION HEADER #{sectionNo + 1}
{_IMAGE_SECTION_HEADER.GetName(&_IMAGE_SECTION_HEADER)} name
       0 physical address
       0x{_IMAGE_SECTION_HEADER.VirtualAddress:X} virtual address
      0x{_IMAGE_SECTION_HEADER.SizeOfRawData:X} size of raw data
     0x{_IMAGE_SECTION_HEADER.PointerToRawData:X} file pointer to raw data (0x{_IMAGE_SECTION_HEADER.PointerToRawData:X} to 0x{_IMAGE_SECTION_HEADER.PointerToRawData + _IMAGE_SECTION_HEADER.SizeOfRawData:X})
       0x{_IMAGE_SECTION_HEADER.PointerToRelocations:X} file pointer to relocation table
       0x{_IMAGE_SECTION_HEADER.PointerToLinenumbers:X} file pointer to line numbers
       {_IMAGE_SECTION_HEADER.NumberOfRelocations} number of relocations
       {_IMAGE_SECTION_HEADER.NumberOfLinenumbers} number of line numbers
  0x{(DWORD)_IMAGE_SECTION_HEADER.Characteristics:X} flags
         {_IMAGE_SECTION_HEADER.Characteristics.ToString()}");
            }

            SECTION_TABLE.Add($"SECT{sectionNo + 1}", new Section(
                $"SECT{sectionNo + 1}",
                _IMAGE_SECTION_HEADER,
                null));
        }

        // Seek to the very end of the file to fetch the string table.
        //      String table is used for names longer than 8 bytes.
        if (isFull) {
            Console.WriteLine("\nCOFF STRING TABLE\n");
        }
        fseek(file, _IMAGE_FILE_HEADER.PointerToSymbolTable
            + (uint)sizeof(_IMAGE_SYMBOL) * _IMAGE_FILE_HEADER.NumberOfSymbols, SeekOrigin.Begin);

        DWORD len = 0;

        cc = fread(&len, (uint)sizeof(DWORD), 1, file);
        if (cc != sizeof(DWORD)) {
            throw new BadImageFormatException();
        }

        len -= sizeof(DWORD);

        BYTE[] bytes = new BYTE[len];
        fixed (void* ptr = bytes) {
            cc = fread(ptr, (DWORD)bytes.Length, sizeof(BYTE), file);
            if (cc != (DWORD)bytes.Length * sizeof(BYTE)) {
                throw new BadImageFormatException();
            }
        }

        Dictionary<DWORD, string> STRING_TABLE = new Dictionary<DWORD, string>();

        for (int i = 0, start = 0; i < len; i++) {
            if (bytes[i] == '\0') {
                DWORD offset = sizeof(DWORD) + (DWORD)start;
                string str = Encoding.UTF8.GetString(bytes, start, i - start);
                if (isFull) {
                    Console.WriteLine($"0x{offset:X6} {str}");
                }
                start = i + 1;
                STRING_TABLE[offset] = str;
            }
        }

        Dictionary<string, DWORD> FUNC_TABLE = new Dictionary<string, DWORD>();

        Debug.Assert(sizeof(_IMAGE_SYMBOL) == 18, "sizeof(_IMAGE_SYMBOL) != 18");

        if (_IMAGE_FILE_HEADER.NumberOfSymbols > 0) {

            if (isFull) {
                Console.WriteLine("\nCOFF SYMBOL TABLE\n");
            }

            fseek(file, _IMAGE_FILE_HEADER.PointerToSymbolTable, SeekOrigin.Begin);

            _IMAGE_SYMBOL _IMAGE_SYMBOL = new _IMAGE_SYMBOL();

            for (int symbolNo = 0; symbolNo < _IMAGE_FILE_HEADER.NumberOfSymbols; symbolNo++) {
                cc = fread(&_IMAGE_SYMBOL, (uint)sizeof(_IMAGE_SYMBOL), 1, file);
                if (cc != sizeof(_IMAGE_SYMBOL)) {
                    throw new BadImageFormatException();
                }

                if (_IMAGE_SYMBOL.SectionNumber == 0 ||
                            _IMAGE_SYMBOL.StorageClass == 0) {
                    continue;
                }

                string name;
                if (*(DWORD*)&_IMAGE_SYMBOL.Name[0] == 0) {
                    name = STRING_TABLE[*(DWORD*)&_IMAGE_SYMBOL.Name[4]];
                } else {
                    name = Encoding.UTF8.GetString(_IMAGE_SYMBOL.Name, _IMAGE_SYMBOL.IMAGE_SIZEOF_SHORT_NAME);
                }

                string SectionNumber = _IMAGE_SYMBOL.SectionNumber != 0xFFFF
                    ? $"SECT{_IMAGE_SYMBOL.SectionNumber}"
                    : "ABS";

                string type = _IMAGE_SYMBOL.Type == 0
                    ? "notype"
                    : $"0x{_IMAGE_SYMBOL.Type:X4}";

                if (isFull) {
                    Console.WriteLine($"{symbolNo:X3}\t{_IMAGE_SYMBOL.Value:X8}\t{SectionNumber}\t{type}\t{_IMAGE_SYMBOL.StorageClass}\t| {name}");
                }

                // Available function
                if (_IMAGE_SYMBOL.StorageClass == IMAGE_SYM_CLASS.EXTERN
                        && _IMAGE_SYMBOL.SectionNumber != 0xFFFF) {

                    Debug.Assert(_IMAGE_SYMBOL.Type == 0 || _IMAGE_SYMBOL.Type == 0x20);

                    Section section = SECTION_TABLE[SectionNumber];

                    section.FUNC_TABLE.Add(_IMAGE_SYMBOL.Value, name);
                }
            }

        }

        // Examine the raw data

        for (int sectionNo = 0; sectionNo < sections.Length; sectionNo++) {
            _IMAGE_SECTION_HEADER _IMAGE_SECTION_HEADER = sections[sectionNo];
            if (_IMAGE_SECTION_HEADER.SizeOfRawData > 0) {
                fseek(file, _IMAGE_SECTION_HEADER.PointerToRawData, SeekOrigin.Begin);
                int ALIGNMENT = 16;
                bytes = new byte[((_IMAGE_SECTION_HEADER.SizeOfRawData + (ALIGNMENT - 1)) & (~(ALIGNMENT - 1)))];
                fixed (void* ptr = bytes) {
                    cc = fread(ptr, _IMAGE_SECTION_HEADER.SizeOfRawData, 1, file);
                    if (cc != _IMAGE_SECTION_HEADER.SizeOfRawData) {
                        throw new BadImageFormatException();
                    }
                }
                if (isFull) Console.WriteLine($"\nRAW DATA #{sectionNo + 1} ({_IMAGE_SECTION_HEADER.GetName(&_IMAGE_SECTION_HEADER)})\n");
                for (int b = 0; b < bytes.Length; b++) {
                    if (b % 16 == 0) {
                        if (isFull) Console.Write($"  {b:X8}: ");
                    }
                    if (b >= _IMAGE_SECTION_HEADER.SizeOfRawData) {
                        if (isFull) Console.Write($"   ");
                    } else {
                        if (isFull) Console.Write($"{bytes[b]:X2} ");
                    }
                    if ((b + 1) % ALIGNMENT == 0) {
                        char[] chars = new char[ALIGNMENT];
                        for (int i = 0; i < chars.Length; i++) {
                            var val = bytes[b - ALIGNMENT + i + 1];
                            if (isprint(val) && !isspace(val)) {
                                chars[i] = (char)val;
                            } else {
                                chars[i] = '.';
                            }
                        }
                        if (isFull) Console.WriteLine("  " + new string(chars));
                    }
                }
                if (isFull) Console.WriteLine();
            }
        }

        // Print byte code

        for (int sectionNo = 0; sectionNo < sections.Length; sectionNo++) {
            _IMAGE_SECTION_HEADER _IMAGE_SECTION_HEADER = sections[sectionNo];
            if (_IMAGE_SECTION_HEADER.SizeOfRawData > 0 && _IMAGE_SECTION_HEADER.IsContentCode()) {

                fseek(file, _IMAGE_SECTION_HEADER.PointerToRawData, SeekOrigin.Begin);

                bytes = new byte[_IMAGE_SECTION_HEADER.SizeOfRawData];
                fixed (void* ptr = bytes) {
                    cc = fread(ptr, _IMAGE_SECTION_HEADER.SizeOfRawData, 1, file);
                    if (cc != _IMAGE_SECTION_HEADER.SizeOfRawData) {
                        throw new BadImageFormatException();
                    }
                }

                Section section = SECTION_TABLE[$"SECT{sectionNo + 1}"];

                if (isFull) Console.WriteLine($"\nCODE #{sectionNo + 1} ({_IMAGE_SECTION_HEADER.GetName(&_IMAGE_SECTION_HEADER)})\n");
                int paras = 0;
                for (DWORD address = 0; address < bytes.Length; address++) {
                    if (section.FUNC_TABLE.TryGetValue(address, out var funcName)) {
                        if (address > 0) {
                            Console.WriteLine("\n};\n");
                            paras--;
                        }
                        Console.Write("static byte[] " + funcName + " = new byte[] {");
                        Console.WriteLine("\n");
                        paras++;
                    }
                    if ((address) % 16 == 0) {
                        Console.Write("\t");
                    }
                    if (address >= _IMAGE_SECTION_HEADER.SizeOfRawData) {
                        Console.Write($"   ");
                    } else {
                        Console.Write($"0x{bytes[address]:X2}");
                        if (address < bytes.Length - 1) {
                            Console.Write($", ");
                        }
                    }
                    if ((address + 1) % 16 == 0) {
                        Console.WriteLine();
                    }
                }
                if (paras > 0) {
                    Console.WriteLine("\n};");
                    paras--;
                }
                Console.WriteLine();
            }
        }
    }

    static bool isprint(byte b) { return b >= 32 && b <= 126; }
    static bool isspace(byte b) { return b >= 9 && b <= 13; }
}
