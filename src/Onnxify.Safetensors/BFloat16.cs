using System.Runtime.InteropServices;

namespace Onnxify.Safetensors;

/// <summary>Represents the 16-bit bfloat payload stored by Safetensors.</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct BFloat16 : IEquatable<BFloat16>
{
    private readonly ushort _value;

    private BFloat16(ushort value)
    {
        _value = value;
    }

    /// <summary>Gets the encoded 16-bit payload.</summary>
    public ushort Value => _value;

    /// <summary>Creates a bfloat16 value from its encoded payload.</summary>
    public static BFloat16 FromEncoded(ushort value) => new(value);

    /// <summary>Expands this bfloat16 payload to its exactly representable single-precision value.</summary>
    public float ToSingle() => Float32Bits.ToSingle(unchecked((int)((uint)_value << 16)));

    public bool Equals(BFloat16 other) => _value == other._value;

    public override bool Equals(object? obj) => obj is BFloat16 other && Equals(other);

    public override int GetHashCode() => _value.GetHashCode();
}
