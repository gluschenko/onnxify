using System.Text.Json;

namespace Onnxify.Safetensors;

/// <summary>
/// Represents a safetensors element type in the managed port and preserves the upstream wire-level data type set.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>Dtype</c>.
/// </remarks>
public enum DataType
{
    /// <summary>Boolean values stored as one byte per element.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::BOOL</c>.</remarks>
    Bool = 1,
    /// <summary>Four-bit floating-point values packed into bytes.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F4</c>.</remarks>
    F4 = 2,
    /// <summary>Six-bit floating-point values using a 2-bit exponent and 3-bit mantissa layout.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F6_E2M3</c>.</remarks>
    F6E2M3 = 3,
    /// <summary>Six-bit floating-point values using a 3-bit exponent and 2-bit mantissa layout.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F6_E3M2</c>.</remarks>
    F6E3M2 = 4,
    /// <summary>Unsigned 8-bit integer values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::U8</c>.</remarks>
    U8 = 5,
    /// <summary>Signed 8-bit integer values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::I8</c>.</remarks>
    I8 = 6,
    /// <summary>Float8 values with E5M2 layout.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F8_E5M2</c>.</remarks>
    F8E5M2 = 7,
    /// <summary>Float8 values with E4M3 layout.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F8_E4M3</c>.</remarks>
    F8E4M3 = 8,
    /// <summary>Float8 values with E8M0 layout.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F8_E8M0</c>.</remarks>
    F8E8M0 = 9,
    /// <summary>Float8 values with E4M3FNUZ layout.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F8_E4M3FNUZ</c>.</remarks>
    F8E4M3Fnuz = 10,
    /// <summary>Float8 values with E5M2FNUZ layout.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F8_E5M2FNUZ</c>.</remarks>
    F8E5M2Fnuz = 11,
    /// <summary>Signed 16-bit integer values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::I16</c>.</remarks>
    I16 = 12,
    /// <summary>Unsigned 16-bit integer values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::U16</c>.</remarks>
    U16 = 13,
    /// <summary>IEEE half-precision floating-point values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F16</c>.</remarks>
    F16 = 14,
    /// <summary>BFloat16 floating-point values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::BF16</c>.</remarks>
    Bf16 = 15,
    /// <summary>Signed 32-bit integer values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::I32</c>.</remarks>
    I32 = 16,
    /// <summary>Unsigned 32-bit integer values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::U32</c>.</remarks>
    U32 = 17,
    /// <summary>IEEE single-precision floating-point values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F32</c>.</remarks>
    F32 = 18,
    /// <summary>Complex values backed by two 32-bit floating-point components.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::C64</c>.</remarks>
    C64 = 19,
    /// <summary>IEEE double-precision floating-point values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::F64</c>.</remarks>
    F64 = 20,
    /// <summary>Signed 64-bit integer values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::I64</c>.</remarks>
    I64 = 21,
    /// <summary>Unsigned 64-bit integer values.</summary>
    /// <remarks>Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>. Original Rust entity: <c>Dtype::U64</c>.</remarks>
    U64 = 22,
}

/// <summary>
/// Provides helpers for translating <see cref="DataType"/> values to and from safetensors wire conventions.
/// </summary>
/// <remarks>
/// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
/// Original Rust entity: <c>impl Dtype</c>.
/// </remarks>
public static class DataTypeExtensions
{
    /// <summary>
    /// Returns the logical number of bits occupied by a single element of the specified data type.
    /// </summary>
    /// <param name="dtype">The safetensors data type to inspect.</param>
    /// <returns>The upstream-compatible element size in bits.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Dtype::bitsize</c>.
    /// </remarks>
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

    /// <summary>
    /// Converts a managed <see cref="DataType"/> into the exact token used by the safetensors header format.
    /// </summary>
    /// <param name="dtype">The data type to serialize.</param>
    /// <returns>The wire name expected in the JSON header.</returns>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Display for Dtype</c>.
    /// </remarks>
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

    /// <summary>
    /// Parses a safetensors header <c>dtype</c> token into the managed <see cref="DataType"/> equivalent.
    /// </summary>
    /// <param name="name">The exact wire token read from the header JSON.</param>
    /// <returns>The matching managed data type.</returns>
    /// <exception cref="JsonException">Thrown when the token is not a known safetensors data type.</exception>
    /// <remarks>
    /// Original Rust file: <c>third_party/safetensors/safetensors/src/tensor.rs</c>.
    /// Original Rust entity: <c>Deserialize for Dtype</c>.
    /// </remarks>
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
