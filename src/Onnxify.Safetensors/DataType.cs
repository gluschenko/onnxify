using System.Text.Json;

namespace Onnxify.Safetensors;

public enum DataType
{
    Bool = 1,
    F4 = 2,
    F6E2M3 = 3,
    F6E3M2 = 4,
    U8 = 5,
    I8 = 6,
    F8E5M2 = 7,
    F8E4M3 = 8,
    F8E8M0 = 9,
    F8E4M3Fnuz = 10,
    F8E5M2Fnuz = 11,
    I16 = 12,
    U16 = 13,
    F16 = 14,
    Bf16 = 15,
    I32 = 16,
    U32 = 17,
    F32 = 18,
    C64 = 19,
    F64 = 20,
    I64 = 21,
    U64 = 22,
}

public static class DtypeExtensions
{
    public static int Bitsize(this DataType dtype)
    {
        return dtype switch
        {
            DataType.F4 => 4,
            DataType.F6E2M3 => 6,
            DataType.F6E3M2 => 6,
            DataType.Bool => 8,
            DataType.U8 => 8,
            DataType.I8 => 8,
            DataType.F8E5M2 => 8,
            DataType.F8E4M3 => 8,
            DataType.F8E8M0 => 8,
            DataType.F8E4M3Fnuz => 8,
            DataType.F8E5M2Fnuz => 8,
            DataType.I16 => 16,
            DataType.U16 => 16,
            DataType.F16 => 16,
            DataType.Bf16 => 16,
            DataType.I32 => 32,
            DataType.U32 => 32,
            DataType.F32 => 32,
            DataType.C64 => 64,
            DataType.F64 => 64,
            DataType.I64 => 64,
            DataType.U64 => 64,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null),
        };
    }

    public static string ToWireName(this DataType dtype)
    {
        return dtype switch
        {
            DataType.Bool => "BOOL",
            DataType.F4 => "F4",
            DataType.F6E2M3 => "F6_E2M3",
            DataType.F6E3M2 => "F6_E3M2",
            DataType.U8 => "U8",
            DataType.I8 => "I8",
            DataType.F8E5M2 => "F8_E5M2",
            DataType.F8E4M3 => "F8_E4M3",
            DataType.F8E8M0 => "F8_E8M0",
            DataType.F8E4M3Fnuz => "F8_E4M3FNUZ",
            DataType.F8E5M2Fnuz => "F8_E5M2FNUZ",
            DataType.I16 => "I16",
            DataType.U16 => "U16",
            DataType.F16 => "F16",
            DataType.Bf16 => "BF16",
            DataType.I32 => "I32",
            DataType.U32 => "U32",
            DataType.F32 => "F32",
            DataType.C64 => "C64",
            DataType.F64 => "F64",
            DataType.I64 => "I64",
            DataType.U64 => "U64",
            _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null),
        };
    }

    public static DataType ParseWireName(string name)
    {
        ArgumentNullException.ThrowIfNull(name);

        return name switch
        {
            "BOOL" => DataType.Bool,
            "F4" => DataType.F4,
            "F6_E2M3" => DataType.F6E2M3,
            "F6_E3M2" => DataType.F6E3M2,
            "U8" => DataType.U8,
            "I8" => DataType.I8,
            "F8_E5M2" => DataType.F8E5M2,
            "F8_E4M3" => DataType.F8E4M3,
            "F8_E8M0" => DataType.F8E8M0,
            "F8_E4M3FNUZ" => DataType.F8E4M3Fnuz,
            "F8_E5M2FNUZ" => DataType.F8E5M2Fnuz,
            "I16" => DataType.I16,
            "U16" => DataType.U16,
            "F16" => DataType.F16,
            "BF16" => DataType.Bf16,
            "I32" => DataType.I32,
            "U32" => DataType.U32,
            "F32" => DataType.F32,
            "C64" => DataType.C64,
            "F64" => DataType.F64,
            "I64" => DataType.I64,
            "U64" => DataType.U64,
            _ => throw new JsonException($"Unknown safetensors dtype '{name}'."),
        };
    }
}
